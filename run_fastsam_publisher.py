"""Run Fast SAM 3D Body's SMPL publisher (run_publisher.py) on a RealSense D555 with this project's asset paths.

The upstream RealSense source hard-codes 848x480 color, which the D555 does not offer (its color profiles are
1280x800/1280x720/896x504/640x360/...), so the default size is patched here before the publisher starts.
Gravity is calibrated from the D555 IMU at startup (keep the camera still), so camera pitch needs no setting.

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
run_publisher.main()
