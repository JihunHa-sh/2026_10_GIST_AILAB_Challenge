"""Camera teleop with locomotion: the operator walks, the robot walks; the upper body follows as VR 3-point targets.

Reads run_fastsam_publisher.py's stream (tcp://localhost:5556, topic "pose") and drives SONIC in planner mode
(`run_sonic.py --input-type zmq_manager --zmq-port 5559`), the way SONIC's PICO teleop does in PLANNER_VR_3PT mode:
  * legs:  SONIC's locomotion planner. Its command comes from the operator's pelvis track (`root_pos_w`, camera at the
           origin, gravity-aligned): velocity in the operator's body frame (forward/left) -> the same direction in the
           robot's frame; the operator's heading change since calibration -> the robot's facing.
  * upper body: SMPL pose -> PICO-style joint poses (vr3pt.py) -> PICO's ThreePointPose (calibrated against the
           robot's measured joints from SONIC's g1_debug output, port 5557).
  * hands: the publisher's Dex3 joint targets, passed through.

Keys (type in this terminal, then ENTER):
    ENTER   start (first time) / recalibrate the arms. Stand at attention (arms hanging straight), facing the
            camera: that pose is mapped to the robot standing at attention (--calib-pose).
    l       locomotion on/off (off: the robot stands, the upper body still follows)
    o       stop SONIC (same as `O` in its terminal)
Log: `[loco]` lines every 0.5 s with the measured pelvis position/velocity and the command sent.

    python 2026_10_GIST_AILAB_Challenge/teleop_planner.py            # env_isaaclab
"""

import argparse
import collections
import json
import math
import os
import select
import sys
import time

import numpy as np
import zmq
from scipy.spatial.transform import Rotation as R

HERE = os.path.dirname(os.path.abspath(__file__))
GR00T = os.path.join(os.path.expanduser(os.environ.get("THIRD_PARTY", "~/third_party")), "GR00T-WholeBodyControl")
sys.path.insert(0, GR00T)
sys.path.insert(0, HERE)
import gear_sonic.scripts.pico_manager_thread_server as pico  # noqa: E402
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message, build_planner_message  # noqa: E402

from vr3pt import ATTENTION_Q, smpl_to_body_poses  # noqa: E402

IDLE, SLOW_WALK = 0, 1  # LocomotionMode

parser = argparse.ArgumentParser()
parser.add_argument("--pose-addr", default="tcp://localhost:5556", help="run_fastsam_publisher.py stream")
parser.add_argument("--addr", default="tcp://*:5559", help="where SONIC (zmq_manager) listens")
parser.add_argument("--rate", type=float, default=50.0)
parser.add_argument("--vel-window", type=float, default=0.8, help="seconds of pelvis track used for the velocity fit")
parser.add_argument("--walk-start", type=float, default=0.20, help="start walking above this speed [m/s]")
parser.add_argument("--walk-stop", type=float, default=0.10, help="stop walking below this speed [m/s]")
parser.add_argument("--speed-gain", type=float, default=1.0, help="robot speed = gain * operator speed")
parser.add_argument("--max-speed", type=float, default=0.6, help="SLOW_WALK speed limit [m/s] (planner: 0.1 .. 0.6)")
parser.add_argument("--turn-deadband", type=float, default=10.0, help="ignore heading changes below this [deg]")
parser.add_argument("--yaw-alpha", type=float, default=0.3, help="EMA factor for the operator heading (per new frame)")
parser.add_argument("--no-loco", action="store_true", help="start with locomotion off")
parser.add_argument("--calib-pose", choices=["attention", "measured"], default="attention",
                    help="robot pose the operator's start pose is mapped to: attention (arms hanging straight, "
                         "operator stands at attention) or the robot's measured pose at that moment")
args = parser.parse_args()


def decode(msg: bytes) -> dict:
    header = json.loads(msg[4:1284].rstrip(b"\0"))
    dt = {"i64": "<i8", "f64": "<f8", "f32": "<f4", "i32": "<i4"}
    out, off = {}, 1284
    for f in header["fields"]:
        n = int(np.prod(f["shape"]))
        out[f["name"]] = np.frombuffer(msg, np.dtype(dt[f["dtype"]]), n, off).reshape(f["shape"]).copy()
        off += n * np.dtype(dt[f["dtype"]]).itemsize
    return out


def heading(body_quat_wxyz: np.ndarray) -> float:
    """Yaw [rad] of the operator's forward axis (SONIC's body_quat convention: robot-like, x forward, z up)."""
    w, x, y, z = body_quat_wxyz.reshape(4)
    fwd = R.from_quat([x, y, z, w]).apply([1.0, 0.0, 0.0])
    return math.atan2(fwd[1], fwd[0])


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Locomotion:
    """Operator pelvis track -> planner (mode, movement, facing, speed)."""

    def __init__(self):
        self.track = collections.deque()  # (stamp, x, y)
        self.last_stamp = 0.0
        self.yaw = None  # filtered operator heading (world)
        self.yaw0 = 0.0  # operator heading at calibration
        self.yaw_off = 0.0  # body_quat heading -> pelvis-track frame (the operator faces the camera at calibration)
        self.pos = np.zeros(3)
        self.facing = 0.0  # robot facing command (relative to its start heading)
        self.walking = False
        self.v = np.zeros(2)
        self.flip_t = None

    def calibrate(self):
        self.yaw0 = self.yaw if self.yaw is not None else 0.0
        self.yaw_off = wrap(math.atan2(-self.pos[1], -self.pos[0]) - self.yaw0)  # facing the camera = toward it
        self.facing, self.walking = 0.0, False
        return math.degrees(self.yaw_off)

    def to_cam_deg(self) -> float:
        """Operator heading vs. the direction to the camera (0 = facing the camera)."""
        return math.degrees(wrap(math.atan2(-self.pos[1], -self.pos[0]) - (self.yaw or 0.0) - self.yaw_off))

    def update(self, pose: dict):
        stamp = float(pose["root_stamp"][0])
        if stamp <= self.last_stamp:
            return
        self.last_stamp = stamp
        p = pose["root_pos_w"].reshape(3)
        self.pos = p.copy()
        self.track.append((stamp, p[0], p[1]))
        while self.track and stamp - self.track[0][0] > args.vel_window:
            self.track.popleft()
        if len(self.track) >= 3:
            t = np.array([s[0] for s in self.track]); t -= t.mean()
            xy = np.array([[s[1], s[2]] for s in self.track])
            self.v = (t[:, None] * (xy - xy.mean(axis=0))).sum(axis=0) / max((t * t).sum(), 1e-6)  # LSQ slope
        yaw = heading(pose["body_quat"])
        if self.yaw is None:
            self.yaw = yaw
        d = wrap(yaw - self.yaw)
        if abs(d) > math.radians(90):  # front/back flips of the pose model: ignore unless they persist 1 s
            self.flip_t = self.flip_t or stamp
            if stamp - self.flip_t < 1.0:
                return
        self.flip_t = None
        self.yaw = wrap(self.yaw + args.yaw_alpha * d)
        target = wrap(self.yaw - self.yaw0)
        if abs(wrap(target - self.facing)) > math.radians(args.turn_deadband):
            self.facing = target

    def command(self, enabled: bool):
        speed = float(np.linalg.norm(self.v))
        if not enabled:
            self.walking = False
        elif self.walking and speed < args.walk_stop:
            self.walking = False
        elif not self.walking and speed > args.walk_start:
            self.walking = True
        yaw = (self.yaw if self.yaw is not None else 0.0) + self.yaw_off
        c, s = math.cos(-yaw), math.sin(-yaw)
        v_body = np.array([c * self.v[0] - s * self.v[1], s * self.v[0] + c * self.v[1]])  # operator frame
        facing = [math.cos(self.facing), math.sin(self.facing), 0.0]
        if not self.walking:
            return IDLE, [0.0, 0.0, 0.0], facing, -1.0, v_body
        d = v_body / max(np.linalg.norm(v_body), 1e-6)
        c, s = math.cos(self.facing), math.sin(self.facing)
        move = [c * d[0] - s * d[1], s * d[0] + c * d[1], 0.0]  # same body-frame direction, robot's facing
        return SLOW_WALK, move, facing, float(np.clip(args.speed_gain * speed, 0.1, args.max_speed)), v_body


def main():
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.CONFLATE, 1)  # single-part messages: keep only the newest pose
    sub.setsockopt(zmq.SUBSCRIBE, b"pose")
    sub.connect(args.pose_addr)
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.LINGER, 0)
    pub.bind(args.addr)
    three_point = pico.ThreePointPose(log_prefix="teleop")
    feedback = pico.FeedbackReader()  # SONIC g1_debug (localhost:5557)
    loco = Locomotion()
    started, loco_on, stop = False, not args.no_loco, False
    pose, pose_t, vr = None, 0.0, None
    last_log = 0.0
    print(f"[teleop] reading {args.pose_addr}, SONIC planner on {args.addr}. Stand still (arms down, facing the "
          "camera) and press ENTER to start.", flush=True)

    while True:
        t_loop = time.time()
        if select.select([sys.stdin], [], [], 0)[0]:
            raw = sys.stdin.readline()
            key = raw.strip().lower()
            if raw == "":  # stdin closed (not a terminal): no more keys
                sys.stdin = open(os.devnull)
            elif key == "":
                if args.calib_pose == "attention":
                    q, ref = ATTENTION_Q, "attention"
                else:
                    feedback.poll_feedback()
                    q = feedback.full_body_q_measured
                    ref = "measured" if q is not None else "zero (no g1_debug yet)"
                    q = np.zeros(29) if q is None else np.asarray(q)[:29]
                three_point.reset_with_measured_q(q)
                off = loco.calibrate()
                print(f"[teleop] {'recalibrated' if started else 'START'}: arms calibrated against the robot's "
                      f"{ref} pose, heading zeroed (heading offset {off:+.0f} deg)", flush=True)
                started = True
            elif key == "l":
                loco_on = not loco_on
                print(f"[teleop] locomotion {'ON' if loco_on else 'OFF'}", flush=True)
            elif key == "o":
                stop = True
                print("[teleop] STOP sent", flush=True)
        try:
            while True:
                pose = decode(sub.recv(zmq.NOBLOCK))
                pose_t = time.time()
        except zmq.Again:
            pass
        if pose is not None:
            if "root_pos_w" in pose:
                loco.update(pose)
            vr = three_point.process_smpl_pose(smpl_to_body_poses(pose["smpl_pose"].reshape(21, 3)))
        fresh = pose is not None and time.time() - pose_t < 0.5
        mode, move, facing, speed, v_body = loco.command(loco_on and fresh)
        if started or stop:
            pub.send(build_command_message(start=started and not stop, stop=stop, planner=True))
            kw = {}
            if vr is not None:
                kw.update(vr_3pt_position=vr[:, :3].flatten().tolist(), vr_3pt_orientation=vr[:, 3:].flatten().tolist())
            if pose is not None and "left_hand_joints" in pose:
                kw.update(left_hand_position=pose["left_hand_joints"].flatten().tolist(),
                          right_hand_position=pose["right_hand_joints"].flatten().tolist())
            pub.send(build_planner_message(mode, move, facing, speed=speed, height=-1.0, **kw))
        if time.time() - last_log > 0.5 and pose is not None:
            last_log = time.time()
            p = loco.pos
            to_cam = loco.to_cam_deg()
            print(f"[loco] pos ({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f}) | v world ({loco.v[0]:+.2f},{loco.v[1]:+.2f}) "
                  f"body fwd {v_body[0]:+.2f} left {v_body[1]:+.2f} | yaw {math.degrees(loco.yaw or 0):+6.1f} "
                  f"face-to-cam {to_cam:+6.1f} | cmd {'WALK' if mode else 'IDLE'} speed {speed:+.2f} "
                  f"facing {math.degrees(loco.facing):+6.1f} | {'fresh' if fresh else 'NO POSE'}"
                  f"{'' if started else ' | waiting for ENTER'}", flush=True)
        if stop:
            time.sleep(0.5)
            break
        time.sleep(max(0.0, 1.0 / args.rate - (time.time() - t_loop)))


if __name__ == "__main__":
    main()
