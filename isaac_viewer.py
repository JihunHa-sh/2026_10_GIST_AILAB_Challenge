"""Isaac Sim window that shows the G1 simulated by isaac_sonic_bridge.py (which runs headless).

Rendering in the physics process stalls the 200 Hz control loop for ~7 ms per frame, and SONIC is sensitive to that
jitter (walking trials fell in GUI mode but not headless, and also fell headless with artificial 7 ms stalls). So the
physics stays headless and this separate process mirrors its state: it subscribes to rt/lowstate (joint angles) and
rt/odostate (root pose) (+ rt/dex3/*/state for the fingers) and only renders. Keys pressed here are forwarded to the bridge:
    9  lower the robot onto its feet and release (again: catch it)    7 / 8  lower / raise the gantry
    R  reset the robot onto the gantry
Run (Isaac Lab env) after starting the bridge:
    ./isaaclab.sh -p 2026_10_GIST_AILAB_Challenge/isaac_viewer.py
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--sonic_root", default=os.path.join(os.environ.get("THIRD_PARTY", "~/third_party"),
                                                       "GR00T-WholeBodyControl"))
parser.add_argument("--interface", default="lo")
parser.add_argument("--domain_id", type=int, default=0)
parser.add_argument("--control_port", type=int, default=5558)
parser.add_argument("--fps", type=float, default=30.0)
parser.add_argument("--screenshot", default=None, help="debug: save a viewport PNG here after --screenshot_after s")
parser.add_argument("--screenshot_after", type=float, default=40.0)
parser.add_argument("--hands", choices=["dex3", "none"], default="dex3", help="must match the bridge's --hands")
parser.add_argument("--no_follow", action="store_true", help="keep the camera fixed instead of following the robot")
AppLauncher.add_app_launcher_args(parser)
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
import zmq  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

SONIC_ROOT = os.path.expanduser(args_cli.sonic_root)
sys.path.insert(0, SONIC_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gear_sonic.envs.manager_env.robots.g1 import G1_CYLINDER_MODEL_12_DEX_CFG  # noqa: E402

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber  # noqa: E402
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_, LowState_, OdoState_  # noqa: E402

from g1_dex3 import DEX3_LEFT, DEX3_RIGHT, NUM_DEX3, add_dex3  # noqa: E402

# same motor order as isaac_sonic_bridge.UNITREE_JOINTS (importing the bridge would start a second app)
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
KEYMAP = {"KEY_9": "band", "NUMPAD_9": "band", "KEY_7": "band_down", "NUMPAD_7": "band_down",
          "KEY_8": "band_up", "NUMPAD_8": "band_up", "R": "reset"}


class StateBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.q = None
        self.root = None  # pos (3), quat wxyz (4)
        self.hand = {"left": None, "right": None}
        self.n = 0

    def on_hand(self, side, msg):
        q = np.array([msg.motor_state[i].q for i in range(NUM_DEX3)])
        with self.lock:
            self.hand[side] = q

    def on_lowstate(self, msg):
        q = np.array([msg.motor_state[i].q for i in range(len(UNITREE_JOINTS))])
        with self.lock:
            self.q, self.n = q, self.n + 1

    def on_odo(self, msg):
        root = np.concatenate([np.array(msg.position), np.array(msg.orientation)])
        with self.lock:
            self.root = root

    def get(self):
        with self.lock:
            return (None if self.q is None else self.q.copy()), (None if self.root is None else self.root.copy()), self.n

    def get_hands(self):
        with self.lock:
            return dict(self.hand)


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / args_cli.fps, device=args_cli.device))
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)
    light = sim_utils.DomeLightCfg(intensity=2500.0)
    light.func("/World/light", light)
    cfg = G1_CYLINDER_MODEL_12_DEX_CFG.copy()
    cfg.prim_path = "/World/Robot"
    if args_cli.hands == "dex3":
        add_dex3(cfg, SONIC_ROOT)
    else:
        cfg.spawn.asset_path = os.path.join(SONIC_ROOT, cfg.spawn.asset_path)
    cfg.spawn.rigid_props.disable_gravity = True  # display only: pose is written every frame
    cfg.spawn.activate_contact_sensors = False
    cfg.spawn.collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=False)
    robot = Articulation(cfg)
    sim.reset()
    u2i = np.array([robot.joint_names.index(j) for j in UNITREE_JOINTS])
    h2i = {} if args_cli.hands != "dex3" else {
        side: np.array([robot.joint_names.index(j) for j in joints]) for side, joints in (("left", DEX3_LEFT), ("right", DEX3_RIGHT))}

    ChannelFactoryInitialize(args_cli.domain_id, args_cli.interface)
    buf = StateBuffer()
    sub_state = ChannelSubscriber("rt/lowstate", LowState_)
    sub_state.Init(buf.on_lowstate, 1)
    sub_odo = ChannelSubscriber("rt/odostate", OdoState_)
    sub_odo.Init(buf.on_odo, 1)
    sub_hand = {}
    for side in h2i:
        sub_hand[side] = ChannelSubscriber(f"rt/dex3/{side}/state", HandState_)
        sub_hand[side].Init(lambda msg, side=side: buf.on_hand(side, msg), 1)

    push = zmq.Context.instance().socket(zmq.PUSH)
    push.setsockopt(zmq.LINGER, 0)
    push.connect(f"tcp://127.0.0.1:{args_cli.control_port}")

    import carb
    import omni.appwindow

    def on_key(event, *_):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS and event.input.name in KEYMAP:
            push.send_string(KEYMAP[event.input.name], zmq.NOBLOCK)
            print(f"[viewer] sent '{KEYMAP[event.input.name]}' to the bridge", flush=True)
        return True

    _kb = carb.input.acquire_input_interface().subscribe_to_keyboard_events(  # noqa: F841
        omni.appwindow.get_default_app_window().get_keyboard(), on_key)
    print("[viewer] mirroring rt/lowstate + rt/odostate | keys: 9 lower/catch, 7/8 gantry, R reset", flush=True)

    period = 1.0 / args_cli.fps
    t_start, shot_done = time.time(), False
    last_print, last_n = time.time(), 0
    joint_vel = torch.zeros_like(robot.data.default_joint_vel)
    while simulation_app.is_running():
        t0 = time.time()
        q_u, root, n = buf.get()
        if q_u is not None and root is not None:
            q = robot.data.default_joint_pos.clone()
            q[0, u2i] = torch.tensor(q_u, dtype=torch.float32, device=sim.device)
            for side, qh in buf.get_hands().items():
                if qh is not None:
                    q[0, h2i[side]] = torch.tensor(qh, dtype=torch.float32, device=sim.device)
            robot.write_joint_state_to_sim(q, joint_vel)
            robot.write_root_pose_to_sim(torch.tensor(root, dtype=torch.float32, device=sim.device)[None])
            robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=sim.device))
            if not args_cli.no_follow:
                p = root[:3]
                sim.set_camera_view(eye=(p[0] + 2.5, p[1] + 2.5, 1.5), target=(p[0], p[1], 0.8))
        sim.step(render=True)  # gravity is off and the state is overwritten every frame -> display only
        robot.update(period)
        if args_cli.screenshot and not shot_done and time.time() - t_start > args_cli.screenshot_after:
            from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

            capture_viewport_to_file(get_active_viewport(), args_cli.screenshot)
            shot_done = True
            print(f"[viewer] screenshot -> {args_cli.screenshot}", flush=True)
        if time.time() - last_print > 5.0:
            rate = (n - last_n) / (time.time() - last_print)
            print(f"[viewer] lowstate {rate:.0f} Hz" + ("" if root is not None else " | waiting for rt/odostate ..."),
                  flush=True)
            last_print, last_n = time.time(), n
        time.sleep(max(0.0, period - (time.time() - t0)))


if __name__ == "__main__":
    main()
    os._exit(0)
