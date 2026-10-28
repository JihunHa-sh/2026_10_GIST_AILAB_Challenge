"""Launch SONIC's C++ deployment (g1_deploy_onnx_ref) the way `deploy.sh sim` does, without `just`/sudo setup.

It runs the binary in a pseudo-terminal so its keyboard controls keep working, and can also inject keys on a
schedule for unattended tests.

    # interactive (type ']' to start the policy, see SONIC docs for the other keys)
    python 2026_10_GIST_AILAB_Challenge/run_sonic.py --input-type zmq
    # scripted: start the policy 1 s after init, enable ZMQ streaming 20 s later
    python 2026_10_GIST_AILAB_Challenge/run_sonic.py --input-type zmq --keys "1:],21:ENTER" --duration 60

Defaults mirror deploy.sh: decoder/encoder policy/release/model_*.onnx, obs config
policy/release/observation_config.yaml, planner V2, reference/example/, output all, interface lo, CRC check off.
"""

import argparse
import os
import pty
import select
import subprocess
import sys
import termios
import time
import tty

THIRD_PARTY = os.path.expanduser(os.environ.get("THIRD_PARTY", "~/third_party"))
SONIC = os.path.join(THIRD_PARTY, "GR00T-WholeBodyControl/gear_sonic_deploy")

parser = argparse.ArgumentParser()
parser.add_argument("--interface", default="lo")
parser.add_argument("--checkpoint", default="policy/release/model", help="prefix of *_encoder.onnx / *_decoder.onnx")
parser.add_argument("--obs-config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config/obs_config_fastsam.yaml"),
                    help="SONIC observation config (default: Fast SAM 3D Body variant without the wrist requirement)")
parser.add_argument("--planner", default="planner/target_vel/V2/planner_sonic.onnx")
parser.add_argument("--motion-data", default="reference/example/")
parser.add_argument("--input-type", default="zmq", help="keyboard | zmq | manager | zmq_manager | ...")
parser.add_argument("--output-type", default="all")
parser.add_argument("--zmq-host", default="localhost")
parser.add_argument("--keys", default="",
                    help='scripted keys "t1:k1,t2:k2", seconds after the binary prints "Init Done"; k may be ENTER')
parser.add_argument("--duration", type=float, default=0.0, help="kill after N s (0 = until it exits)")
parser.add_argument("extra", nargs=argparse.REMAINDER, help="extra args passed to the binary")
args = parser.parse_args()

env = dict(os.environ)
libs = [
    f"{THIRD_PARTY}/TensorRT-10.13.3.9/lib",
    f"{THIRD_PARTY}/onnxruntime-linux-x64-1.16.3/lib",
    f"{THIRD_PARTY}/sonic_deps/lib",
    f"{SONIC}/thirdparty/unitree_sdk2/thirdparty/lib/x86_64",
    "/usr/local/cuda-12.2/lib64",
]
env["LD_LIBRARY_PATH"] = ":".join(libs + [env.get("LD_LIBRARY_PATH", "")])

cmd = [
    os.path.join(SONIC, "target/release/g1_deploy_onnx_ref"), args.interface,
    f"{args.checkpoint}_decoder.onnx", args.motion_data,
    "--obs-config", args.obs_config,
    "--encoder-file", f"{args.checkpoint}_encoder.onnx",
    "--planner-file", args.planner,
    "--input-type", args.input_type,
    "--output-type", args.output_type,
    "--zmq-host", args.zmq_host,
    "--disable-crc-check",
] + [a for a in args.extra if a != "--"]

SPECIAL_KEYS = {"ENTER": b"\n", "SPACE": b" ", "ESC": b"\x1b"}
schedule = []
for item in filter(None, args.keys.split(",")):
    t, k = item.split(":", 1)
    schedule.append((float(t), SPECIAL_KEYS.get(k, k.encode())))

print("[run_sonic]", " ".join(cmd), flush=True)
master, slave = pty.openpty()
proc = subprocess.Popen(cmd, cwd=SONIC, env=env, stdin=slave, stdout=slave, stderr=slave, close_fds=True)
os.close(slave)

interactive = sys.stdin.isatty()
old_attr = None
if interactive:
    old_attr = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
t0 = time.time()
t_init = None  # keys are timed from "Init Done" (model/TRT loading time varies run to run)
tail = b""
try:
    while proc.poll() is None:
        now = time.time() - t0
        while schedule and t_init is not None and time.time() - t_init >= schedule[0][0]:
            os.write(master, schedule.pop(0)[1])
        if args.duration and now > args.duration:
            break
        fds = [master] + ([sys.stdin] if interactive else [])
        r, _, _ = select.select(fds, [], [], 0.05)
        if master in r:
            try:
                data = os.read(master, 4096)
            except OSError:
                break
            sys.stdout.buffer.write(data)
            sys.stdout.flush()
            if t_init is None:
                tail = (tail + data)[-256:]
                if b"Init Done" in tail:
                    t_init = time.time()
        if interactive and sys.stdin in r:
            os.write(master, os.read(sys.stdin.fileno(), 64))
finally:
    if old_attr is not None:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_attr)
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
