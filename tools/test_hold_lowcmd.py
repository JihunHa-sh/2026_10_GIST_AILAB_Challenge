"""Bridge self-test without SONIC: publish rt/lowcmd that PD-holds SONIC's standing pose and report rt/lowstate.

    python 2026_10_GIST_AILAB_Challenge/tools/test_hold_lowcmd.py --seconds 12
(run in the Isaac Lab env while isaac_sonic_bridge.py is running)
"""

import argparse
import threading
import time

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=12.0)
parser.add_argument("--interface", default="lo")
args = parser.parse_args()

# SONIC's default standing pose (g1.py init_state) in Unitree motor order
STAND = np.zeros(29)
STAND[[0, 6]] = -0.312   # hip pitch
STAND[[3, 9]] = 0.669    # knee
STAND[[4, 10]] = -0.363  # ankle pitch
STAND[[15, 22]] = 0.2    # shoulder pitch
STAND[16], STAND[23] = 0.2, -0.2  # shoulder roll
STAND[[18, 25]] = 0.6    # elbow
KP = np.array([150, 150, 150, 200, 40, 40] * 2 + [200, 40, 40] + [40, 40, 40, 40, 20, 20, 20] * 2, dtype=float)
KD = np.array([2, 2, 2, 4, 2, 2] * 2 + [4, 2, 2] + [1, 1, 1, 1, 0.5, 0.5, 0.5] * 2, dtype=float)

ChannelFactoryInitialize(0, args.interface)
pub = ChannelPublisher("rt/lowcmd", LowCmd_)
pub.Init()
latest = {"msg": None, "n": 0}
lock = threading.Lock()


def on_state(msg):
    with lock:
        latest["msg"], latest["n"] = msg, latest["n"] + 1


sub = ChannelSubscriber("rt/lowstate", LowState_)
sub.Init(on_state, 1)

cmd = unitree_hg_msg_dds__LowCmd_()
t0, last = time.time(), 0.0
errs = []
while time.time() - t0 < args.seconds:
    for i in range(29):
        m = cmd.motor_cmd[i]
        m.mode, m.q, m.dq, m.kp, m.kd, m.tau = 1, float(STAND[i]), 0.0, float(KP[i]), float(KD[i]), 0.0
    pub.Write(cmd)
    with lock:
        msg, n = latest["msg"], latest["n"]
    if msg is not None:
        q = np.array([msg.motor_state[i].q for i in range(29)])
        errs.append(np.abs(q - STAND).max())
        if time.time() - t0 - last > 1.0:
            last = time.time() - t0
            quat = np.array(msg.imu_state.quaternion)
            print(f"[test] t={last:4.1f}s lowstate msgs {n} | max |q-stand| {errs[-1]:.3f} rad | imu quat {quat.round(3)}"
                  f" | tick {msg.tick}", flush=True)
    time.sleep(0.002)
if latest["msg"] is not None:
    q = np.array([latest["msg"].motor_state[i].q for i in range(29)])
    worst = np.argsort(-np.abs(q - STAND))[:6]
    print("[test] largest joint errors (unitree idx: q / target):",
          ", ".join(f"{i}: {q[i]:+.2f}/{STAND[i]:+.2f}" for i in worst))
print(f"[test] done: lowstate received {latest['n']} | final max joint error {errs[-1] if errs else float('nan'):.3f} rad")
