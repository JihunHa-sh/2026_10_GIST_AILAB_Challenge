"""SMPL body pose (Fast SAM 3D Body / any SMPL source) -> SONIC VR 3-point targets (left wrist, right wrist, neck).

SONIC's planner mode takes the upper body as a VR 3-point pose, exactly what its PICO teleop sends
(gear_sonic/scripts/pico_manager_thread_server.py, StreamMode.PLANNER_VR_3PT). PICO delivers 24 SMPL joints as
global poses; `compute_from_body_poses` there turns them into SMPL parameters via ``global * Ry(180)`` and parent-
relative rotations. `smpl_to_body_poses` is the inverse, so PICO's own `ThreePointPose` (frame conversion, per-joint
offsets, calibration against the robot's measured pose) can be used unchanged. Everything ends up relative to the
pelvis, so the root orientation does not matter and only the 21 local joint rotations are needed.
Import after putting GR00T-WholeBodyControl on sys.path.
"""

import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

GR00T = os.path.join(os.path.expanduser(os.environ.get("THIRD_PARTY", "~/third_party")), "GR00T-WholeBodyControl")
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
# SMPL hand joints (22, 23) sit ~9 cm past the wrists along the forearm in the neutral model (rest frame, metres)
HAND_OFFSET = {22: np.array([0.087, -0.010, -0.015]), 23: np.array([-0.087, -0.010, -0.015])}
_RY180 = R.from_euler("y", 180, degrees=True)
# Robot reference for the arm calibration (ThreePointPose.reset_with_measured_q): standing at attention, arms hanging
# almost straight (G1's elbow is bent 90 deg at q=0 and straight at ~1.57), shoulders slightly abducted to clear the
# torso. Unitree motor order. The operator stands at attention when calibrating.
ATTENTION_Q = np.zeros(29)
ATTENTION_Q[[16, 18]] = [0.15, 1.5]    # left shoulder roll, left elbow
ATTENTION_Q[[23, 25]] = [-0.15, 1.5]   # right shoulder roll, right elbow


def smpl_to_body_poses(smpl_pose: np.ndarray) -> np.ndarray:
    """(21, 3) local axis-angles of SMPL joints 1..21 -> (24, 7) PICO-style global poses [x, y, z, qx, qy, qz, qw]."""
    from gear_sonic.trl.utils.torch_transform import compute_human_joints

    pose = np.asarray(smpl_pose, dtype=np.float64).reshape(21, 3)
    joints = compute_human_joints(
        body_pose=torch.as_tensor(pose.reshape(1, 63), dtype=torch.float32),
        global_orient=torch.zeros(1, 3),
        human_joints_info_path=os.path.join(GR00T, "gear_sonic/data/human/human_joints_info.pkl"),
    )[0, :22].numpy().astype(np.float64)
    local = [R.identity()] + [R.from_rotvec(v) for v in pose] + [R.identity(), R.identity()]
    glob = []
    for i, p in enumerate(SMPL_PARENTS):
        glob.append(local[i] if p < 0 else glob[p] * local[i])
    pos = np.zeros((24, 3))
    pos[:22] = joints
    for i, off in HAND_OFFSET.items():
        pos[i] = pos[SMPL_PARENTS[i]] + glob[i].apply(off)
    out = np.zeros((24, 7))
    out[:, :3] = pos
    for i in range(24):
        out[i, 3:] = (glob[i] * _RY180).as_quat()  # scalar-last, as PICO
    return out
