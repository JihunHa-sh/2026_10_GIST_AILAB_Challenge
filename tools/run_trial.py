"""Unattended end-to-end trial: Isaac bridge + SONIC deploy (+ optional SMPL file stream), with a fall verdict.

The bridge exits first (--seconds), so every logged state was produced while SONIC was still in control.
A trial counts as a fall if the pelvis drops below --fall_z at any time after the band was released.

    python 2026_10_GIST_AILAB_Challenge/tools/run_trial.py --tag t1 --seconds 60
    python 2026_10_GIST_AILAB_Challenge/tools/run_trial.py --tag w1 --seconds 90 --stream walk --enter_after 20
"""

import argparse
import os
import re
import signal
import subprocess
import sys
import time

HOME = os.path.expanduser("~")
THIRD_PARTY = os.path.expanduser(os.environ.get("THIRD_PARTY", "~/third_party"))
SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # this project folder


def find_isaaclab(start: str) -> str:
    """The Isaac Lab checkout this folder lives in (contains isaaclab.sh); override with $ISAACLAB_PATH."""
    if os.environ.get("ISAACLAB_PATH"):
        return os.environ["ISAACLAB_PATH"]
    d = start
    while d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, "isaaclab.sh")):
            return d
        d = os.path.dirname(d)
    return os.path.join(HOME, "IsaacLab")


ISAACLAB = find_isaaclab(SCRIPTS)
STREAMS = {
    "walk": os.path.join(THIRD_PARTY, "GR00T-WholeBodyControl/sample_data/smpl_filtered/walk_forward_amateur_001__A001.pkl"),
}

parser = argparse.ArgumentParser()
parser.add_argument("--tag", required=True)
parser.add_argument("--seconds", type=float, default=60.0, help="bridge sim seconds")
parser.add_argument("--release_after", type=float, default=6.0)
parser.add_argument("--stream", default=None, help="SMPL file or key of STREAMS to stream over ZMQ")
parser.add_argument("--enter_after", type=float, default=20.0, help="press ENTER (ZMQ streaming) N s after init")
parser.add_argument("--hold_first", type=float, default=60.0,
                    help="stream the first (standing) frame this long before playing, so streaming is enabled while "
                         "the 'operator' stands still, as in real use")
parser.add_argument("--gui", action="store_true", help="run the bridge with the Isaac Sim window")
parser.add_argument("--viewer", action="store_true", help="also run isaac_viewer.py (bridge stays headless)")
parser.add_argument("--publisher", action="store_true", help="use the live D555 Fast SAM 3D Body publisher as the source")
parser.add_argument("--planner", action="store_true",
                    help="source = tools/planner_test.py (SONIC --input-type zmq_manager, locomotion planner)")
parser.add_argument("--planner_args", default="", help="extra args for planner_test.py")
parser.add_argument("--stream_args", default="", help="extra args for stream_smpl_file.py, e.g. '--hand-cycle 3'")
parser.add_argument("--fall_z", type=float, default=0.5)
parser.add_argument("bridge_args", nargs=argparse.REMAINDER)
args = parser.parse_args()

log = f"/tmp/sonic_trial_{args.tag}"
procs = []
# the GUI viewer's startup (shader compilation etc.) starves the physics process -> start it first and wait
if args.viewer:
    procs.append(subprocess.Popen(["nice", "-n", "10", os.path.join(ISAACLAB, "isaaclab.sh"), "-p",
                                   os.path.join(SCRIPTS, "isaac_viewer.py"),
                                   "--screenshot", log + "_viewer.png", "--screenshot_after", str(args.seconds + 5)],
                                  cwd=ISAACLAB, stdout=open(log + "_viewer.log", "w"), stderr=subprocess.STDOUT,
                                  env=dict(os.environ, PYTHONUNBUFFERED="1"), start_new_session=True))
if args.viewer:
    while "[viewer] mirroring" not in open(log + "_viewer.log").read():
        time.sleep(1)
    time.sleep(5)
bridge = subprocess.Popen(
    [os.path.join(ISAACLAB, "isaaclab.sh"), "-p", os.path.join(SCRIPTS, "isaac_sonic_bridge.py")] + ([] if args.gui else ["--headless"]) + [
     "--release_band_after", str(args.release_after), "--max_seconds", str(args.seconds)]
    + [a for a in args.bridge_args if a != "--"],
    cwd=ISAACLAB, stdout=open(log + "_bridge.log", "w"), stderr=subprocess.STDOUT,
    env=dict(os.environ, PYTHONUNBUFFERED="1"), start_new_session=True)
while "motors mapped" not in open(log + "_bridge.log").read():
    if bridge.poll() is not None:
        sys.exit(f"bridge exited early, see {log}_bridge.log")
    time.sleep(1)

keys = "1:]"
input_type = "zmq"
if args.planner:
    keys, input_type = "", "zmq_manager"  # start comes from the 'command' topic
    procs.append(subprocess.Popen(
        [sys.executable, os.path.join(SCRIPTS, "tools/planner_test.py")] + args.planner_args.split(),
        stdout=open(log + "_planner.log", "w"), stderr=subprocess.STDOUT, start_new_session=True))
elif args.publisher:
    keys += f",{args.enter_after}:ENTER"
    fastsam_py = os.path.join(HOME, "miniconda3/envs/fast_sam_3d_body/bin/python")
    nvj = os.path.join(HOME, "miniconda3/envs/fast_sam_3d_body/lib/python3.11/site-packages/nvidia/nvjitlink/lib")
    procs.append(subprocess.Popen([fastsam_py, os.path.join(SCRIPTS, "run_fastsam_publisher.py")],
                                  stdout=open(log + "_publisher.log", "w"), stderr=subprocess.STDOUT,
                                  env=dict(os.environ, LD_LIBRARY_PATH=nvj), start_new_session=True))
elif args.stream:
    keys += f",{args.enter_after}:ENTER"
    procs.append(subprocess.Popen(
        [sys.executable, os.path.join(SCRIPTS, "tools/stream_smpl_file.py"), "--file", STREAMS.get(args.stream, args.stream),
         "--loop", "--hold-first", str(args.hold_first)] + args.stream_args.split(), stdout=open(log + "_stream.log", "w"), stderr=subprocess.STDOUT,
        start_new_session=True))
deploy = subprocess.Popen(
    [sys.executable, os.path.join(SCRIPTS, "run_sonic.py"), "--input-type", input_type] + (["--keys", keys] if keys else []),
    stdin=subprocess.DEVNULL, stdout=open(log + "_deploy.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
bridge.wait()
def kill_group(pgid: int, grace: float = 8.0):
    """SIGTERM the whole session, then SIGKILL whatever is left (Isaac Sim ignores SIGTERM and outlives its wrapper)."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        t_end = time.time() + grace
        while time.time() < t_end:
            try:
                os.killpg(pgid, 0)  # anyone left in the group?
            except ProcessLookupError:
                return
            time.sleep(0.5)


for p in [deploy] + procs + [bridge]:
    kill_group(p.pid)
    p.poll()

text = open(log + "_bridge.log").read()
released = text.find("band released")
states = [(float(t), float(x), float(y), float(z), float(u)) for t, x, y, z, u in re.findall(
    r"t=\s*([\d.]+)s.*?pelvis xy \(([-+\d.]+),([-+\d.]+)\) z ([\d.]+) upright ([-\d.]+)", text[released:] if released >= 0 else "")]
deploy_txt = re.sub(r"\x1b\[[0-9;]*m", "", open(log + "_deploy.log", errors="ignore").read())
streaming = "ZMQ STREAMING MODE: ENABLED" in deploy_txt
errors = sorted(set(re.findall(r"✗ Error: [^\n]+", deploy_txt)))[:3]
if released < 0 or not states:
    verdict = "NO RELEASE (policy never took over?)"
else:
    fell = [s for s in states if s[3] < args.fall_z]
    verdict = f"FELL at t={fell[0][0]:.1f}s" if fell else "STOOD"
    travel = ((states[-1][1] - states[0][1]) ** 2 + (states[-1][2] - states[0][2]) ** 2) ** 0.5
    verdict += f" | min z {min(s[3] for s in states):.3f} | travel {travel:.2f} m | final xy ({states[-1][1]:+.2f},{states[-1][2]:+.2f})"
print(f"[trial {args.tag}] {verdict} | streaming={streaming}" + (f" | deploy errors: {errors}" if errors else ""), flush=True)
