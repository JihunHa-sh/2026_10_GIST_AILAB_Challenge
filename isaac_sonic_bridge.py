"""Isaac Sim stand-in for the Unitree G1 on the DDS bus, so SONIC's C++ deployment can drive it unchanged.

This replaces GR00T-WholeBodyControl's MuJoCo sim loop (gear_sonic/utils/mujoco_sim): it subscribes to
``rt/lowcmd`` and applies the same per-motor law the real robot / MuJoCo bridge use,

    tau = tau_ff + kp * (q_des - q) + kd * (dq_des - dq),

and publishes ``rt/lowstate`` (motors in Unitree order + pelvis IMU), ``rt/secondary_imu`` (torso IMU) and
``rt/odostate``. The robot is SONIC's own training asset (URDF, armature, effort/velocity limits from
gear_sonic/envs/manager_env/robots/g1.py) with the joint drives zeroed, so all torque comes from the command.

Like the MuJoCo sim, the pelvis starts on an elastic band (a virtual gantry). Keys in the Isaac Sim window:
    9  lower the robot onto its feet and release (again: catch it)      7 / 8  lower / raise the band
    R  reset the robot onto the band
Run (Isaac Lab env), then start SONIC with `deploy.sh`-equivalent args on interface `lo`:
    ./isaaclab.sh -p 2026_10_GIST_AILAB_Challenge/isaac_sonic_bridge.py
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--sonic_root", default=os.path.join(os.environ.get("THIRD_PARTY", "~/third_party"),
                                                       "GR00T-WholeBodyControl"))
parser.add_argument("--interface", default="lo", help="DDS network interface (lo for sim)")
parser.add_argument("--domain_id", type=int, default=0)
parser.add_argument("--physics_hz", type=int, default=200,
                    help="physics + lowstate rate. 200 Hz = SONIC's training sim rate and keeps the GUI real-time; "
                         "use 500 (real G1 lowstate rate) headless or with --pd explicit, which oscillates at 200 Hz")
parser.add_argument("--pd", choices=["implicit", "explicit"], default="implicit",
                    help="implicit: kp/kd/q/dq from lowcmd go into PhysX joint drives (how SONIC was trained in Isaac Lab); "
                         "explicit: torque = tau + kp*(q*-q) + kd*(dq*-dq) per physics step (MuJoCo bridge law)")
parser.add_argument("--render_hz", type=int, default=50)
parser.add_argument("--band_height", type=float, default=1.0, help="elastic band anchor height for the pelvis [m]")
parser.add_argument("--band_fade", type=float, default=0.0, help="seconds to fade the band out after touchdown (0 = cut)")
parser.add_argument("--release_band_after", type=float, default=0.0,
                    help="drop the band automatically N s after the first non-zero-gain lowcmd (0 = manual, key 9)")
parser.add_argument("--max_seconds", type=float, default=0.0, help="stop after N s of sim time (0 = run forever)")
parser.add_argument("--debug_joints", action="store_true", help="print per-joint command/state at each log")
parser.add_argument("--control_port", type=int, default=5558,
                    help="ZMQ port for band/reset commands from isaac_viewer.py (keys pressed in the viewer window)")
parser.add_argument("--debug_stall_ms", type=float, default=0.0, help="debug: sleep this long on every render step")
parser.add_argument("--log_every", type=float, default=2.0, help="status print period [s]")
AppLauncher.add_app_launcher_args(parser)
# single robot: CPU PhysX is far cheaper per step than the GPU pipeline
parser.set_defaults(device="cpu")
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

SONIC_ROOT = os.path.expanduser(args_cli.sonic_root)
sys.path.insert(0, SONIC_ROOT)
from gear_sonic.envs.manager_env.robots.g1 import G1_CYLINDER_MODEL_12_DEX_CFG  # noqa: E402

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber  # noqa: E402
from unitree_sdk2py.idl.default import (  # noqa: E402
    unitree_hg_msg_dds__IMUState_,
    unitree_hg_msg_dds__LowState_,
    unitree_hg_msg_dds__OdoState_,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_, LowCmd_, LowState_, OdoState_  # noqa: E402

# Unitree motor order (= gear_sonic_deploy/g1/g1_29dof.xml actuator order = LowCmd/LowState index)
UNITREE_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
NUM_MOTORS = len(UNITREE_JOINTS)


class ElasticBand:
    """Virtual gantry on the pelvis; same gains as unitree_mujoco / gear_sonic's MuJoCo sim.

    ``lower()`` starts a soft landing: the anchor descends at ``lower_speed`` until the pelvis reaches standing
    height (feet loaded), then the band lets go. Dropping the hanging robot from ~0.2 m (MuJoCo's key 9) lands it
    in whatever pose the policy happens to hold mid-air, which is hit-or-miss.
    """

    def __init__(self, height: float, release_height: float = 0.80, lower_speed: float = 0.3, fade_time: float = 1.5):
        self.kp_pos, self.kd_pos, self.kp_ang, self.kd_ang = 10000.0, 1000.0, 1000.0, 10.0
        self.point = np.array([0.0, 0.0, height])
        self.height0 = height
        self.length = 0.0
        self.enable = True
        self.lowering = False
        self.release_height, self.lower_speed, self.fade_time = release_height, lower_speed, fade_time
        self.scale = 1.0  # gain multiplier, faded to 0 once the feet carry the robot
        self.fading = False

    def lower(self):
        if self.enable:
            self.lowering = True

    def catch(self):
        self.enable, self.lowering, self.fading, self.scale, self.length = True, False, False, 1.0, 0.0

    def update(self, dt: float, pelvis_z: float):
        if self.enable and self.lowering and not self.fading:
            self.length -= self.lower_speed * dt
            if pelvis_z <= self.release_height or self.height0 + self.length <= self.release_height - 0.05:
                self.fading = True  # feet down: hand the weight over gradually
        if self.enable and self.fading:
            self.scale -= dt / self.fade_time
            if self.scale <= 0.0:
                self.enable, self.lowering, self.fading, self.scale = False, False, False, 0.0
                return True  # fully released this step
        return False

    def wrench(self, pos, quat_wxyz, lin_vel, ang_vel):
        f = self.kp_pos * (self.point - pos + np.array([0.0, 0.0, self.length])) - self.kd_pos * lin_vel
        rotvec = Rotation.from_quat(np.roll(quat_wxyz, -1)).as_rotvec()
        tq = -self.kp_ang * rotvec - self.kd_ang * ang_vel
        return f * self.scale, tq * self.scale


class LowCmdBuffer:
    """Latest rt/lowcmd, converted to arrays in Unitree order."""

    def __init__(self):
        self.lock = threading.Lock()
        self.q = np.zeros(NUM_MOTORS)
        self.dq = np.zeros(NUM_MOTORS)
        self.kp = np.zeros(NUM_MOTORS)
        self.kd = np.zeros(NUM_MOTORS)
        self.tau = np.zeros(NUM_MOTORS)
        self.count = 0

    def handler(self, msg: LowCmd_):
        mc = msg.motor_cmd
        with self.lock:
            for i in range(NUM_MOTORS):
                self.q[i], self.dq[i] = mc[i].q, mc[i].dq
                self.kp[i], self.kd[i], self.tau[i] = mc[i].kp, mc[i].kd, mc[i].tau
            self.count += 1

    def snapshot(self):
        with self.lock:
            return self.q.copy(), self.dq.copy(), self.kp.copy(), self.kd.copy(), self.tau.copy(), self.count


def make_robot_cfg():
    cfg = G1_CYLINDER_MODEL_12_DEX_CFG.copy()
    cfg.prim_path = "/World/Robot"
    cfg.spawn.asset_path = os.path.join(SONIC_ROOT, cfg.spawn.asset_path)
    # all torque comes from the lowcmd PD law -> zero the simulator-side drives (keep armature and limits)
    for act in cfg.actuators.values():
        act.stiffness = 0.0
        act.damping = 0.0
    cfg.init_state.pos = (0.0, 0.0, args_cli.band_height)
    return cfg


KEYMAP = {"KEY_9": "band", "NUMPAD_9": "band", "KEY_7": "band_down", "NUMPAD_7": "band_down",
          "KEY_8": "band_up", "NUMPAD_8": "band_up", "R": "reset"}


def setup_keyboard(on_command):
    """Keys in an Isaac Sim window -> on_command(name) (used by the bridge in GUI mode and by isaac_viewer.py)."""
    import carb
    import omni.appwindow

    def on_key(event, *_):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS and event.input.name in KEYMAP:
            on_command(KEYMAP[event.input.name])
        return True

    appwindow = omni.appwindow.get_default_app_window()
    inp = carb.input.acquire_input_interface()
    return inp.subscribe_to_keyboard_events(appwindow.get_keyboard(), on_key)


def main():
    if not args_cli.headless:
        # Kit's own frame-rate limiter paces rendering to wall time too; together with our real-time pacing below
        # the two sleeps add up and the sim runs at ~0.5x real time (SONIC then sees a sluggish robot).
        import carb

        carb.settings.get_settings().set_bool("/app/runLoops/main/rateLimitEnabled", False)
    dt = 1.0 / args_cli.physics_hz
    render_every = max(int(round(args_cli.physics_hz / args_cli.render_hz)), 1)
    # contact material exactly as in SONIC training (modular_tracking_env_cfg: terrain material, also the sim default):
    # friction 1.0 combined by multiply. Isaac's defaults (0.5) make feet slip while walking.
    material = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0,
                                              friction_combine_mode="multiply", restitution_combine_mode="multiply")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, render_interval=render_every,
                                                              device=args_cli.device, physics_material=material))
    sim.set_camera_view(eye=(2.5, 2.5, 1.5), target=(0.0, 0.0, 0.8))
    ground = sim_utils.GroundPlaneCfg(physics_material=material)
    ground.func("/World/ground", ground)
    sim_utils.DomeLightCfg(intensity=2500.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2500.0))
    robot = Articulation(make_robot_cfg())
    sim.reset()

    names = robot.joint_names
    missing = [j for j in UNITREE_JOINTS if j not in names]
    assert not missing, f"joints missing in the asset: {missing}"
    u2i = np.array([names.index(j) for j in UNITREE_JOINTS])  # unitree index -> isaac joint index
    pelvis = robot.body_names.index("pelvis")
    torso = robot.body_names.index("torso_link")
    print(f"[bridge] {NUM_MOTORS} motors mapped | physics {args_cli.physics_hz} Hz | render every {render_every}")

    # ---- DDS ----
    ChannelFactoryInitialize(args_cli.domain_id, args_cli.interface)
    low_state = unitree_hg_msg_dds__LowState_()
    torso_imu = unitree_hg_msg_dds__IMUState_()
    odo = unitree_hg_msg_dds__OdoState_()
    pub_state = ChannelPublisher("rt/lowstate", LowState_)
    pub_state.Init()
    pub_imu = ChannelPublisher("rt/secondary_imu", IMUState_)
    pub_imu.Init()
    pub_odo = ChannelPublisher("rt/odostate", OdoState_)
    pub_odo.Init()
    cmd = LowCmdBuffer()
    sub_cmd = ChannelSubscriber("rt/lowcmd", LowCmd_)
    sub_cmd.Init(cmd.handler, 1)

    band = ElasticBand(args_cli.band_height, fade_time=max(args_cli.band_fade, 1e-3))
    state = {"reset": False}

    def reset_robot():
        root = robot.data.default_root_state.clone()
        root[:, 2] = args_cli.band_height
        robot.write_root_state_to_sim(root)
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
        band.catch()
        robot.reset()

    # remote keys from isaac_viewer.py (the physics bridge runs headless so rendering never stalls it)
    import zmq

    ctl = zmq.Context.instance().socket(zmq.PULL)
    ctl.bind(f"tcp://127.0.0.1:{args_cli.control_port}")

    def handle_key(key: str):
        if key == "band":
            if band.enable:
                band.lower()
                print("[bridge] lowering the robot onto its feet ...", flush=True)
            else:
                band.catch()
                print("[bridge] elastic band ON (robot caught)", flush=True)
        elif key == "band_up":
            band.length += 0.1
        elif key == "band_down":
            band.length -= 0.1
        elif key == "reset":
            state["reset"] = True

    _kb = None if args_cli.headless else setup_keyboard(handle_key)  # noqa: F841 (keep the subscription alive)
    reset_robot()
    prev_vel_w = np.zeros(3)
    sim_time, step = 0.0, 0
    first_active_cmd_t = None
    t_wall0 = time.time()
    last_log = 0.0
    zeros_t = torch.zeros(1, 1, 3, device=sim.device)
    gains_state = {"kp": np.zeros(NUM_MOTORS), "kd": np.zeros(NUM_MOTORS)}
    perf = {"physics": 0.0, "render": 0.0, "n_physics": 0, "n_render": 0}

    while simulation_app.is_running():
        while True:
            try:
                handle_key(ctl.recv_string(zmq.NOBLOCK))
            except zmq.Again:
                break
        if state["reset"]:
            state["reset"] = False
            reset_robot()
            print("[bridge] robot reset onto the band")

        # ---- command -> joint torques (Unitree order -> Isaac order) ----
        q_des, dq_des, kp, kd, tau_ff, n_cmd = cmd.snapshot()
        q = robot.data.joint_pos[0].cpu().numpy()[u2i]
        dq = robot.data.joint_vel[0].cpu().numpy()[u2i]
        if args_cli.pd == "explicit":
            tau_u = tau_ff + kp * (q_des - q) + kd * (dq_des - dq)
            tau = np.zeros(len(names))
            tau[u2i] = tau_u
            robot.set_joint_effort_target(torch.tensor(tau, dtype=torch.float32, device=sim.device)[None])
        else:
            def to_isaac(x):
                out = np.zeros(len(names))
                out[u2i] = x
                return torch.tensor(out, dtype=torch.float32, device=sim.device)[None]

            if not np.array_equal(kp, gains_state["kp"]) or not np.array_equal(kd, gains_state["kd"]):
                robot.write_joint_stiffness_to_sim(to_isaac(kp))
                robot.write_joint_damping_to_sim(to_isaac(kd))
                gains_state["kp"], gains_state["kd"] = kp.copy(), kd.copy()
            robot.set_joint_position_target(to_isaac(q_des))
            robot.set_joint_velocity_target(to_isaac(dq_des))
            robot.set_joint_effort_target(to_isaac(tau_ff))

        if first_active_cmd_t is None and n_cmd > 0 and np.any(kp > 0):
            first_active_cmd_t = sim_time
            print(f"[bridge] receiving lowcmd with gains at t={sim_time:.2f}s")
        if (args_cli.release_band_after > 0 and band.enable and first_active_cmd_t is not None
                and sim_time - first_active_cmd_t > args_cli.release_band_after):
            if not band.lowering:
                band.lower()
                print(f"[bridge] lowering the robot automatically at t={sim_time:.2f}s")

        # ---- elastic band on the pelvis ----
        root_pos = robot.data.root_pos_w[0].cpu().numpy()
        root_quat = robot.data.root_quat_w[0].cpu().numpy()  # w, x, y, z
        root_lin_w = robot.data.root_lin_vel_w[0].cpu().numpy()
        root_ang_w = robot.data.root_ang_vel_w[0].cpu().numpy()
        if band.update(dt, float(root_pos[2])):
            print(f"[bridge] feet down, band released at t={sim_time:.2f}s (pelvis z {root_pos[2]:.3f})")
        if band.enable:
            f, tq = band.wrench(root_pos, root_quat, root_lin_w, root_ang_w)
            robot.set_external_force_and_torque(
                torch.tensor(f, dtype=torch.float32, device=sim.device).view(1, 1, 3),
                torch.tensor(tq, dtype=torch.float32, device=sim.device).view(1, 1, 3),
                body_ids=[pelvis], is_global=True)
        else:
            robot.set_external_force_and_torque(zeros_t, zeros_t, body_ids=[pelvis], is_global=True)

        robot.write_data_to_sim()
        do_render = step % render_every == 0
        t_s = time.perf_counter()
        sim.step(render=do_render)
        if do_render and args_cli.debug_stall_ms > 0:
            time.sleep(args_cli.debug_stall_ms / 1e3)
        perf["render" if do_render else "physics"] += time.perf_counter() - t_s
        perf["n_render" if do_render else "n_physics"] += 1
        robot.update(dt)
        sim_time += dt
        step += 1

        # ---- state -> rt/lowstate, rt/secondary_imu, rt/odostate ----
        q = robot.data.joint_pos[0].cpu().numpy()[u2i]
        dq = robot.data.joint_vel[0].cpu().numpy()[u2i]
        ddq = robot.data.joint_acc[0].cpu().numpy()[u2i]
        tau_est = robot.data.applied_torque[0].cpu().numpy()[u2i]
        for i in range(NUM_MOTORS):
            ms = low_state.motor_state[i]
            ms.q, ms.dq, ms.ddq, ms.tau_est = float(q[i]), float(dq[i]), float(ddq[i]), float(tau_est[i])
        root_quat = robot.data.root_quat_w[0].cpu().numpy()
        vel_w = robot.data.root_lin_vel_w[0].cpu().numpy()
        gyro_b = robot.data.root_ang_vel_b[0].cpu().numpy()  # IMU gyro is body-frame (MuJoCo qvel[3:6])
        acc_w = (vel_w - prev_vel_w) / dt
        prev_vel_w = vel_w
        low_state.imu_state.quaternion[:] = root_quat.tolist()
        low_state.imu_state.gyroscope[:] = gyro_b.tolist()
        low_state.imu_state.accelerometer[:] = acc_w.tolist()
        low_state.tick = int(sim_time * 1e3)

        tq_w = robot.data.body_link_quat_w[0, torso]
        torso_ang_b = math_utils.quat_apply_inverse(tq_w[None], robot.data.body_link_ang_vel_w[0, torso][None])[0]
        torso_imu.quaternion[:] = tq_w.cpu().numpy().tolist()
        torso_imu.gyroscope[:] = torso_ang_b.cpu().numpy().tolist()

        odo.position[:] = robot.data.root_pos_w[0].cpu().numpy().tolist()
        odo.linear_velocity[:] = vel_w.tolist()
        odo.orientation[:] = root_quat.tolist()
        odo.angular_velocity[:] = gyro_b.tolist()
        odo.tick = low_state.tick

        pub_state.Write(low_state)
        pub_imu.Write(torso_imu)
        pub_odo.Write(odo)

        if sim_time - last_log >= args_cli.log_every:
            last_log = sim_time
            up = math_utils.quat_apply(robot.data.root_quat_w, torch.tensor([[0.0, 0.0, 1.0]], device=sim.device))[0]
            rtf = sim_time / max(time.time() - t_wall0, 1e-6)
            rtf_now = args_cli.log_every / max(time.time() - state.get("t_log", t_wall0 - args_cli.log_every), 1e-6)
            state["t_log"] = time.time()
            if rtf_now < 0.95 and first_active_cmd_t is not None:
                print(f"[bridge] WARNING: simulation is running at {rtf_now:.2f}x real time; SONIC needs ~1.0x. "
                      "Close other heavy apps / run isaac_viewer.py with `nice -n 10`.", flush=True)
            print(f"[bridge] t={sim_time:6.2f}s rtf={rtf:4.2f} | lowcmd msgs {n_cmd} | band {'on ' if band.enable else 'off'}"
                  f" | pelvis xy ({root_pos[0]:+.2f},{root_pos[1]:+.2f}) z {root_pos[2]:.3f} upright {up[2].item():.3f}"
                  f" | ms/step physics {1e3 * perf['physics'] / max(perf['n_physics'], 1):.2f}"
                  f" render {1e3 * perf['render'] / max(perf['n_render'], 1):.1f}"
                  f" | physx time {sim.current_time:.2f}s", flush=True)
            if args_cli.debug_joints and n_cmd > 0:
                err = q_des - q
                print("[bridge]   q_des-q (unitree order): " + " ".join(f"{e:+.2f}" for e in err))
                print("[bridge]   kp: " + " ".join(f"{k:.0f}" for k in kp) + " | kd: " + " ".join(f"{k:.1f}" for k in kd))
                print("[bridge]   q: " + " ".join(f"{v:+.2f}" for v in q))
        if args_cli.max_seconds and sim_time >= args_cli.max_seconds:
            break

        # real-time pacing: SONIC runs on the wall clock
        ahead = sim_time - (time.time() - t_wall0)
        if ahead > 0:
            time.sleep(ahead)


if __name__ == "__main__":
    main()
    os._exit(0)  # app.close() can hang after DDS threads are started
