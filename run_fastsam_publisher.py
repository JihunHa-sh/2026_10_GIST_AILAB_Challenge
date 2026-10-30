"""Run Fast SAM 3D Body's SMPL publisher (run_publisher.py) on a RealSense D555 with this project's asset paths.

The upstream RealSense source hard-codes 848x480 color, which the D555 does not offer (its color profiles are
1280x800/1280x720/896x504/640x360/...), so the default size is patched here before the publisher starts.
Gravity is calibrated from the D555 IMU at startup (keep the camera still), so camera pitch needs no setting.
Hands: finger curl from the model's hand keypoints is sent as Dex3 joint targets (`left_hand_joints` /
`right_hand_joints`, which SONIC forwards to rt/dex3/*/cmd); see hand_retarget.py. --no-hands sends the body only.
Each message also carries the pelvis position `root_pos_w` (gravity-aligned world frame, camera at the origin) and
`root_stamp` for teleop_planner.py; SONIC ignores both.

Run in the `fast_sam_3d_body` conda env:
    python 2026_10_GIST_AILAB_Challenge/run_fastsam_publisher.py            # add --record to save video + SMPL
"""

import argparse
import os
import sys

THIRD_PARTY = os.path.expanduser(os.environ.get("THIRD_PARTY", "~/third_party"))
FASTSAM = os.path.join(THIRD_PARTY, "Fast-SAM-3D-Body")
HERE = os.path.dirname(os.path.abspath(__file__))
MHR_ASSETS = os.path.join(THIRD_PARTY, "mhr/tools/mhr_smpl_conversion/assets")

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--width", type=int, default=896)
parser.add_argument("--height", type=int, default=504)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--no-hands", action="store_true", help="do not send Dex3 hand joints")
parser.add_argument("--hand-open-deg", type=float, nargs=2, default=(40.0, 50.0), metavar=("THUMB", "FINGERS"),
                    help="summed finger bend [deg] that counts as fully open")
parser.add_argument("--hand-closed-deg", type=float, nargs=2, default=(110.0, 160.0), metavar=("THUMB", "FINGERS"),
                    help="summed finger bend [deg] that counts as fully closed")
parser.add_argument("--hand-mode", choices=["grab", "fingers"], default="grab",
                    help="grab: each hand fully closes above --grab-close curl and opens below --grab-open; "
                         "fingers: proportional per-finger curl")
parser.add_argument("--grab-close", type=float, default=0.5, help="mean index/middle curl (0..1) that closes the hand")
parser.add_argument("--grab-open", type=float, default=0.3, help="curl below which a closed hand opens again")
parser.add_argument("--thumb", choices=["fingers", "own"], default="fingers",
                    help="Dex3 thumb follows the index/middle curl (default) or the model's own thumb keypoints")
parser.add_argument("--root-log", type=float, default=0.0, help="print the pelvis position every N s (0 = off)")
parser.add_argument("--hand-log", type=float, default=2.0, help="print finger bend/curl every N s (0 = off)")
ours, rest = parser.parse_known_args()

sys.path.insert(0, FASTSAM)
os.chdir(FASTSAM)  # upstream resolves checkpoints/ and mhr2smpl/ relative to the repo

defaults = {
    "--source": "camera",
    "--smpl-model-path": os.path.join(HERE, "assets/smpl/SMPL_NEUTRAL.pkl"),
    "--nn-model-dir": os.path.join(FASTSAM, "mhr2smpl/experiments/multiview_n30000_e500"),
    "--mhr2smpl-mapping-path": os.path.join(MHR_ASSETS, "mhr2smpl_mapping.npz"),
    "--mhr-mesh-path": os.path.join(MHR_ASSETS, "mhr_face_mask.ply"),
    "--smoother-dir": os.path.join(FASTSAM, "mhr2smpl/experiments/smoother_w5"),
}
# YOLO-Pose: TensorRT engine if build_tensorrt.sh has been run, else the PyTorch weights
_engine = os.path.join(FASTSAM, "checkpoints/yolo/yolo11m-pose.engine")
defaults["--yolo-model"] = _engine if os.path.exists(_engine) else os.path.join(FASTSAM, "checkpoints/yolo/yolo11m-pose.pt")
argv = list(rest)
for k, v in defaults.items():
    if k not in argv:
        argv += [k, v]
sys.argv = [os.path.join(FASTSAM, "run_publisher.py")] + argv
print("[fastsam] camera", f"{ours.width}x{ours.height}@{ours.fps}", "|", " ".join(argv), flush=True)

import run_publisher  # noqa: E402

_create = run_publisher.create_video_source


def create_video_source(source_type, **kwargs):
    # upstream calls create_video_source("camera", width=848, height=480, fps=30)
    if source_type == "camera":
        kwargs.update(width=ours.width, height=ours.height, fps=ours.fps)
    return _create(source_type, **kwargs)


run_publisher.create_video_source = create_video_source

import threading  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from sam_3d_body import SAM3DBodyEstimator  # noqa: E402

sys.path.insert(0, HERE)
from hand_retarget import Dex3Retargeter  # noqa: E402

hands = None if ours.no_hands else Dex3Retargeter(
    open_deg=tuple(ours.hand_open_deg), closed_deg=tuple(ours.hand_closed_deg), thumb_mode=ours.thumb,
    mode=ours.hand_mode, grab_close=ours.grab_close, grab_open=ours.grab_open)
# pelvis position in a gravity-aligned frame for teleop_planner.py: camera at the origin, x = camera forward (level),
# y = left, z = up. pred_cam_t / keypoints are OpenCV camera coordinates, while upstream's R_world_cam already includes
# a fixed axis swap (R_zup_adjustment) meant for the body orientation, so it is undone here.
R_ZUP_ADJ = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])  # run_publisher.py, symmetric
root = {"cam": None, "world": np.zeros(3), "stamp": 0.0, "R_world_cam": None}
lock = threading.Lock()
hand_log = {"t": 0.0}
_process = SAM3DBodyEstimator.process_one_image
_prepare = run_publisher.RealtimePublisher._prepare_publish_pose


def process_one_image(self, *args, **kwargs):
    t = time.time()
    outputs = _process(self, *args, **kwargs)
    if len(outputs) == 1:  # the publisher only uses single-person frames
        out = outputs[0]
        hips = np.asarray(out["pred_keypoints_3d"])[[9, 10]].mean(axis=0)  # MHR70 left/right hip
        with lock:
            root["cam"], root["stamp"] = hips + np.asarray(out["pred_cam_t"]).reshape(3), t
            if root["R_world_cam"] is not None:  # v_cam = R_world_cam @ v_world (gravity_alignment.py)
                root["world"] = root["R_world_cam"].T @ (R_ZUP_ADJ @ root["cam"])
            if hands is not None:
                hands.update(out["pred_keypoints_3d"])
        if ours.root_log > 0 and time.time() - hand_log.get("root", 0.0) > ours.root_log:
            hand_log["root"] = time.time()
            print(f"[root] cam_t {np.round(np.asarray(out['pred_cam_t']).reshape(3), 2)} hips_kp {np.round(hips, 2)} "
                  f"world {np.round(root['world'], 2)}", flush=True)
        if hands is not None and ours.hand_log > 0 and time.time() - hand_log["t"] > ours.hand_log:
            hand_log["t"] = time.time()
            for side in ("left", "right"):
                b, c = hands.last_bend[side], hands.curl[side]
                print(f"[hands] {side:5s} bend deg " + " ".join(f"{f} {b[f]:5.0f}" for f in b)
                      + " | curl " + " ".join(f"{f} {c[f]:.2f}" for f in c)
                      + (f" | {'GRAB' if hands.grab[side] else 'open'}" if hands.mode == "grab" else ""), flush=True)
    return outputs


def prepare_publish_pose(self, *args, **kwargs):
    root["R_world_cam"] = self.R_world_cam
    return _prepare(self, *args, **kwargs)


SAM3DBodyEstimator.process_one_image = process_one_image
run_publisher.RealtimePublisher._prepare_publish_pose = prepare_publish_pose


class ExtendedZMQPublisher(run_publisher.ZMQPublisher):
    """Upstream message (protocol v2) + SONIC's optional Dex3 fields + the pelvis position (ignored by SONIC)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        fields = [
            {"name": "frame_index", "dtype": "i64", "shape": [1]},
            {"name": "body_quat", "dtype": "f64", "shape": [1, 4]},  # WXYZ format
            {"name": "smpl_joints", "dtype": "f64", "shape": [1, 24, 3]},
            {"name": "smpl_pose", "dtype": "f64", "shape": [1, 21, 3]},
            {"name": "root_pos_w", "dtype": "f64", "shape": [1, 3]},  # pelvis [m]: x camera-forward, y left, z up
            {"name": "root_stamp", "dtype": "f64", "shape": [1]},  # wall time of the frame it came from
        ]
        if hands is not None:
            fields += [{"name": "left_hand_joints", "dtype": "f64", "shape": [1, 7]},
                       {"name": "right_hand_joints", "dtype": "f64", "shape": [1, 7]}]
        self.header_bytes = self._build_header(fields)

    def publish(self, body_quat, smpl_joints, smpl_pose):
        with lock:
            extra = root["world"].astype(np.float64).tobytes() + np.array([root["stamp"]], np.float64).tobytes()
            if hands is not None:
                extra += hands.joints("left").astype(np.float64).tobytes() + hands.joints("right").astype(np.float64).tobytes()
        data = (
            self.topic_bytes
            + self.header_bytes
            + np.array([self.frame_idx], dtype=np.int64).tobytes()
            + np.asarray(body_quat, dtype=np.float64).reshape(1, 4).tobytes()
            + np.asarray(smpl_joints, dtype=np.float64).reshape(1, 24, 3).tobytes()
            + np.asarray(smpl_pose, dtype=np.float64).reshape(1, 21, 3).tobytes()
            + extra
        )
        self.sock.send(data)
        self.frame_idx += 1


run_publisher.ZMQPublisher = ExtendedZMQPublisher

run_publisher.main()
