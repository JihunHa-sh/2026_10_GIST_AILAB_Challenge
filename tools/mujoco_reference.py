"""Reference run of SONIC's *official* MuJoCo sim (gear_sonic run_sim_loop) without a viewer, releasing the
elastic band after a delay and logging pelvis height/uprightness, to compare against isaac_sonic_bridge.py.

Run with GR00T-WholeBodyControl/.venv_sim/bin/python from the GR00T-WholeBodyControl root:
    .venv_sim/bin/python ~/IsaacLab/2026_10_GIST_AILAB_Challenge/tools/mujoco_reference.py --release-after 30 --seconds 45
"""

import argparse
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory, init_channel

parser = argparse.ArgumentParser()
parser.add_argument("--release-after", type=float, default=30.0, help="wall seconds before dropping the band")
parser.add_argument("--seconds", type=float, default=45.0)
args = parser.parse_args()

cfg = SimLoopConfig()
wbc = cfg.load_wbc_yaml()
wbc["ENV_NAME"] = cfg.env_name
wbc["ENABLE_ONSCREEN"] = False
init_channel(config=wbc)
sim = SimulatorFactory.create_simulator(config=wbc, env_name=cfg.env_name, onscreen=False, offscreen=False)
sim.start_as_thread()
env = sim.sim_env
t0 = time.time()
released = False
while time.time() - t0 < args.seconds:
    time.sleep(2.0)
    el = time.time() - t0
    if not released and el > args.release_after and env.elastic_band is not None:
        env.elastic_band.enable = False
        released = True
        print(f"[mujoco] band released at {el:.1f}s", flush=True)
    qpos = env.mj_data.qpos
    up = Rotation.from_quat(np.roll(qpos[3:7], -1)).apply([0, 0, 1])[2]
    n = 0 if env.unitree_bridge is None else int(env.unitree_bridge.low_cmd_received)
    print(f"[mujoco] t={el:5.1f}s pelvis z {qpos[2]:.3f} upright {up:.3f} | lowcmd received {n}", flush=True)
sim._running = False
import os  # noqa: E402

os._exit(0)
