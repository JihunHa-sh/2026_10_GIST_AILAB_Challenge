"""Drive SONIC's locomotion planner (input type zmq_manager) with a fixed command schedule, no camera.

Sends the 'command' topic (start + planner mode) and 'planner' messages at --rate Hz on the same port Fast SAM uses,
with SONIC's own message builders (gear_sonic/utils/teleop/zmq/zmq_planner_sender.py). Default schedule: stand,
walk forward, stand, walk backward, stand, sidestep left, stand, turn left 90 deg in place, stand.
--arms: upper body from a synthetic SMPL arm motion as VR 3-point targets (vr3pt.py + PICO's ThreePointPose,
calibrated against the robot standing at attention (vr3pt.ATTENTION_Q)): calibrate with the arms down, raise
both arms forward, raise the left arm sideways, walk forward with both arms forward, arms down.

    python 2026_10_GIST_AILAB_Challenge/tools/planner_test.py
    python 2026_10_GIST_AILAB_Challenge/run_sonic.py --input-type zmq_manager      # SONIC side
"""

import argparse
import math
import os
import sys
import time

import zmq

GR00T = os.path.join(os.path.expanduser(os.environ.get("THIRD_PARTY", "~/third_party")), "GR00T-WholeBodyControl")
sys.path.insert(0, GR00T)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message, build_planner_message  # noqa: E402

IDLE, SLOW_WALK, WALK = 0, 1, 2  # LocomotionMode (gear_sonic/scripts/pico_manager_thread_server.py)

parser = argparse.ArgumentParser()
parser.add_argument("--addr", default="tcp://*:5556")
parser.add_argument("--rate", type=float, default=50.0)
parser.add_argument("--speed", type=float, default=0.3, help="SLOW_WALK speed [m/s] (0.1 .. 0.6)")
parser.add_argument("--start-delay", type=float, default=0.0, help="send idle commands only for N s first")
parser.add_argument("--segment", type=float, default=6.0, help="seconds per motion segment")
parser.add_argument("--rest", type=float, default=4.0, help="seconds of standing between segments")
parser.add_argument("--arms", action="store_true", help="upper-body VR 3-point demo instead of the walking schedule")
args = parser.parse_args()

# (name, mode, movement direction in the robot's start frame, facing yaw [deg])
SEGMENTS = [("forward", SLOW_WALK, (1, 0), 0), ("backward", SLOW_WALK, (-1, 0), 0),
            ("left", SLOW_WALK, (0, 1), 0), ("turn left 90", IDLE, (0, 0), 90)]
schedule, t = [("stand", IDLE, (0, 0), 0, args.start_delay)], 0.0
yaw = 0
for name, mode, move, dyaw in SEGMENTS:
    yaw += dyaw
    schedule += [(name, mode, move, yaw, args.segment), ("stand", IDLE, (0, 0), yaw, args.rest)]

if args.arms:
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    import gear_sonic.scripts.pico_manager_thread_server as pico  # noqa: E402

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from vr3pt import ATTENTION_Q, smpl_to_body_poses  # noqa: E402

    L_SH, R_SH = 15, 16  # SMPL joints 16/17 (shoulders) in the 21-joint body pose

    def arms_pose(fwd: float, left_up: float) -> np.ndarray:
        """Arms hanging (70 deg down), raised forward by fwd*90 deg and the left arm sideways by left_up*160 deg."""
        pose = np.zeros((21, 3))
        pose[L_SH] = (R.from_euler("x", -90 * fwd, degrees=True) * R.from_euler("z", -70 + 160 * left_up, degrees=True)).as_rotvec()
        pose[R_SH] = (R.from_euler("x", -90 * fwd, degrees=True) * R.from_euler("z", 70, degrees=True)).as_rotvec()
        return pose

    three_point = pico.ThreePointPose()
    feedback = pico.FeedbackReader()  # SONIC's g1_debug output (localhost:5557)
    # (name, mode, movement, facing deg, duration, arm phase function of segment time u in [0, 1])
    ramp = lambda u: 0.5 - 0.5 * math.cos(2 * math.pi * u)  # 0 -> 1 -> 0  # noqa: E731
    schedule = [("stand", IDLE, (0, 0), 0, args.start_delay, None),
                ("calibrate (arms down)", IDLE, (0, 0), 0, 3.0, lambda u: (0, 0)),
                ("both arms forward", IDLE, (0, 0), 0, args.segment, lambda u: (ramp(u), 0)),
                ("left arm sideways up", IDLE, (0, 0), 0, args.segment, lambda u: (0, ramp(u))),
                ("walk forward, arms forward", SLOW_WALK, (1, 0), 0, args.segment, lambda u: (1, 0)),
                ("arms down", IDLE, (0, 0), 0, args.rest, lambda u: (0, 0))]
else:
    schedule = [seg + (None,) for seg in schedule]

sock = zmq.Context.instance().socket(zmq.PUB)
sock.setsockopt(zmq.LINGER, 0)
sock.bind(args.addr)
time.sleep(0.5)  # let SONIC's subscribers connect
print(f"[planner] publishing on {args.addr} at {args.rate:.0f} Hz", flush=True)

t0, last = time.time(), None
for name, mode, move, yaw_deg, dur, arms in schedule:
    if name != last:
        print(f"[planner] {time.time():.1f} t={time.time() - t0:5.1f}s {name} (mode {mode}, move {move}, facing {yaw_deg} deg)", flush=True)
        last = name
    if name.startswith("calibrate"):
        three_point.reset_with_measured_q(ATTENTION_Q)  # operator's arms-down pose = robot at attention
    t_start = time.time()
    t_end = t_start + dur
    while time.time() < t_end:
        y = math.radians(yaw_deg)
        facing = [math.cos(y), math.sin(y), 0.0]
        vr = {}
        if arms is not None:
            vr3 = three_point.process_smpl_pose(smpl_to_body_poses(arms_pose(*arms((time.time() - t_start) / dur))))
            vr = {"vr_3pt_position": vr3[:, :3].flatten().tolist(), "vr_3pt_orientation": vr3[:, 3:].flatten().tolist()}
        sock.send(build_command_message(start=True, stop=False, planner=True))
        sock.send(build_planner_message(mode, [float(move[0]), float(move[1]), 0.0], facing,
                                        speed=args.speed if mode == SLOW_WALK else -1.0, height=-1.0, **vr))
        time.sleep(1.0 / args.rate)
print("[planner] done", flush=True)
