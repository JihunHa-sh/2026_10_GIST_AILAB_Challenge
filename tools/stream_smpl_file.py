"""Stream an SMPL motion file to SONIC over ZMQ Protocol v2, in exactly Fast SAM 3D Body's wire format.

Used to test the SMPL -> SONIC -> Isaac Sim path without the camera/pose model. The SMPL -> (body_quat,
root-local joints) conversion mirrors gear_sonic/scripts/pico_manager_thread_server.py:process_smpl_joints, i.e. the
in-distribution format of SONIC's SMPL encoder; the message layout mirrors Fast-SAM-3D-Body
mocap/realtime/publisher.py (topic "pose", 1280-byte JSON header, frame_index/body_quat/smpl_joints/smpl_pose).

Run in the Isaac Lab env:
    python 2026_10_GIST_AILAB_Challenge/tools/stream_smpl_file.py \
        --file ~/third_party/GR00T-WholeBodyControl/sample_data/smpl_filtered/walk_forward_amateur_001__A001.pkl
"""

import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import torch
import zmq

GR00T = os.path.join(os.path.expanduser(os.environ.get("THIRD_PARTY", "~/third_party")), "GR00T-WholeBodyControl")
parser = argparse.ArgumentParser()
parser.add_argument("--file", required=True)
parser.add_argument("--addr", default="tcp://*:5556")
parser.add_argument("--rate", type=float, default=50.0, help="publish rate [Hz] (Fast SAM publishes at 50)")
parser.add_argument("--loop", action="store_true")
parser.add_argument("--start-delay", type=float, default=0.0)
parser.add_argument("--hold-first", type=float, default=0.0, help="repeat the first frame for N s before playing")
parser.add_argument("--hand-cycle", type=float, default=0.0,
                    help="also send Dex3 hand joints, alternating open/closed every N s (0 = no hand fields)")
args = parser.parse_args()

sys.path.insert(0, GR00T)
os.chdir(GR00T)  # compute_human_joints loads gear_sonic/data/human/human_joints_info.pkl relative to the repo
from gear_sonic.isaac_utils.rotations import remove_smpl_base_rot, smpl_root_ytoz_up  # noqa: E402
from gear_sonic.trl.utils.torch_transform import (  # noqa: E402
    angle_axis_to_quaternion,
    compute_human_joints,
    quat_apply,
    quat_inv,
    quaternion_to_angle_axis,
)


def process_smpl(pose_aa: np.ndarray):
    """pose_aa (T, 72) -> body_quat (T, 4 wxyz), smpl_joints_local (T, 24, 3), smpl_pose (T, 21, 3)."""
    pose = torch.as_tensor(pose_aa, dtype=torch.float32)
    global_orient, body_pose = pose[:, :3], pose[:, 3:]
    q = smpl_root_ytoz_up(angle_axis_to_quaternion(global_orient))
    joints = compute_human_joints(body_pose=body_pose[:, :63], global_orient=quaternion_to_angle_axis(q))
    q = remove_smpl_base_rot(q, w_last=False)
    local = quat_apply(quat_inv(q).unsqueeze(1).repeat(1, joints.shape[1], 1), joints)
    local = local - local[:, :1]  # root-relative, as Fast SAM 3D Body publishes (mocap/utils/pose_protocol.py)
    return q.numpy().astype(np.float64), local[:, :24].numpy().astype(np.float64), \
        body_pose[:, :63].reshape(-1, 21, 3).numpy().astype(np.float64)


d = joblib.load(os.path.expanduser(args.file))
fps = float(d.get("fps", 30.0))
body_quat, smpl_joints, smpl_pose = process_smpl(d["pose_aa"])
T = len(body_quat)
print(f"[stream] {args.file}: {T} frames @ {fps:.0f} fps -> publishing at {args.rate:.0f} Hz on {args.addr}")

ctx = zmq.Context()
sock = ctx.socket(zmq.PUB)
sock.setsockopt(zmq.LINGER, 0)
sock.bind(args.addr)
fields = [
    {"name": "frame_index", "dtype": "i64", "shape": [1]},
    {"name": "body_quat", "dtype": "f64", "shape": [1, 4]},
    {"name": "smpl_joints", "dtype": "f64", "shape": [1, 24, 3]},
    {"name": "smpl_pose", "dtype": "f64", "shape": [1, 21, 3]},
]
if args.hand_cycle > 0:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from hand_retarget import DEX3_CLOSED  # noqa: E402

    fields += [{"name": "left_hand_joints", "dtype": "f64", "shape": [1, 7]},
               {"name": "right_hand_joints", "dtype": "f64", "shape": [1, 7]}]
header = json.dumps({"v": 2, "endian": "le", "count": 1, "fields": fields},
                    separators=(",", ":")).encode().ljust(1280, b"\0")

time.sleep(args.start_delay)
t0, frame_idx, n = time.time(), 0, 0
while True:
    el = time.time() - t0
    src_t = max(el - args.hold_first, 0.0)
    i = int(src_t * fps)
    if i >= T:
        if not args.loop:
            break
        t0, i = time.time() - args.hold_first, 0
    msg = (b"pose" + header + np.array([frame_idx], np.int64).tobytes() + body_quat[i].tobytes()
           + smpl_joints[i].tobytes() + smpl_pose[i].tobytes())
    if args.hand_cycle > 0:
        closed = int(el / args.hand_cycle) % 2 == 1
        msg += b"".join((DEX3_CLOSED[s] * closed).astype(np.float64).tobytes() for s in ("left", "right"))
    sock.send(msg)
    frame_idx += 1
    if frame_idx % int(args.rate * 2) == 0:
        print(f"[stream] t={el:5.1f}s source frame {i}/{T}", flush=True)
    time.sleep(1.0 / args.rate)
print("[stream] done")
