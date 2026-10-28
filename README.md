# 2026_10_GIST_AILAB_Challenge

Whole-body teleoperation of a **Unitree G1 in Isaac Sim** from a single **RealSense D555**, the way
[Fast SAM 3D Body](https://github.com/yangtiming/Fast-SAM-3D-Body) drives the G1 — no IK in the loop:

```
D555 ─► Fast SAM 3D Body ─SMPL (ZMQ :5556, v2)─► SONIC C++ deploy ─DDS rt/lowcmd─► isaac_sonic_bridge.py (headless physics)
        (fast_sam_3d_body env)                    (GR00T-WholeBodyControl) ◄─rt/lowstate──   │ rt/lowstate + rt/odostate
                                                                                              ▼
                                                                         isaac_viewer.py (Isaac Sim window, keys)
```

[SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl) is NVIDIA's learned whole-body controller: its SMPL
encoder maps the human body model (joint rotations included) to the robot and keeps balance, so elbows, waist and
legs follow the operator. Hands are not retargeted.

This folder is meant to be cloned **into an Isaac Lab checkout** (tested with Isaac Lab v2.3.2):
```bash
cd ~/IsaacLab
git clone https://github.com/JihunHa-sh/2026_10_GIST_AILAB_Challenge.git
```
First-time installation of the external pieces (SONIC, Fast SAM 3D Body, TensorRT, ...): see **[SETUP.md](SETUP.md)**.

## Run (four terminals, in this order)

| Terminal | Conda env | Role |
|---|---|---|
| 1 | `env_isaaclab` | Isaac Sim viewer |
| 2 | `env_isaaclab` | physics bridge (headless) |
| 3 | `env_isaaclab` | SONIC controller (keyboard input goes here) |
| 4 | `fast_sam_3d_body` | D555 + Fast SAM 3D Body publisher |

**1. Viewer** — start first and wait until the window is up (its startup is CPU-heavy and slows the physics).
```bash
conda activate env_isaaclab && cd ~/IsaacLab
nice -n 10 ./isaaclab.sh -p 2026_10_GIST_AILAB_Challenge/isaac_viewer.py
```

**2. Physics bridge**
```bash
conda activate env_isaaclab && cd ~/IsaacLab
./isaaclab.sh -p 2026_10_GIST_AILAB_Challenge/isaac_sonic_bridge.py --headless
```
Its log should show `rtf=1.00`. The G1 hangs on a virtual gantry.

**3. SONIC controller**
```bash
conda activate env_isaaclab && cd ~/IsaacLab
python 2026_10_GIST_AILAB_Challenge/run_sonic.py --input-type zmq
```
Wait for `Init Done` (~20 s).

**4. Camera publisher**
```bash
conda activate fast_sam_3d_body && cd ~/IsaacLab
python 2026_10_GIST_AILAB_Challenge/run_fastsam_publisher.py
```
Keep the camera still for the first second (gravity calibration from the D555 IMU, so camera tilt needs no
setting). Stand 2-3 m away, whole body in view, nobody else in the picture (it only publishes when exactly one
person is detected). `Live: ... infer throughput` should read ~22-28 fps.

**Bring-up**
1. Terminal 3: `]` → policy running (robot still on the gantry).
2. Viewer window: `9` → the robot is lowered onto its feet and released.
3. Stand still in a neutral pose, then terminal 3: `ENTER` → streaming on, the robot follows you.
   `ENTER` again pauses (the robot holds); get close to the robot's pose before resuming.
4. Terminal 3: `O` = stop. Viewer: `9` catches the robot, `R` resets it onto the gantry, `7`/`8` move the gantry.

## Files

| File | Purpose |
|---|---|
| `isaac_sonic_bridge.py` | Isaac Sim stand-in for the G1 on DDS (replaces SONIC's MuJoCo sim) |
| `isaac_viewer.py` | Isaac Sim window mirroring the bridge's robot state; forwards keys to the bridge |
| `run_sonic.py` | Launch SONIC's `g1_deploy_onnx_ref` like `deploy.sh sim`, without `just`/sudo |
| `run_fastsam_publisher.py` | Fast SAM 3D Body publisher on the D555 (896x504; upstream's 848x480 is not a D555 mode) |
| `config/obs_config_fastsam.yaml` | SONIC obs config for SMPL streaming (from Fast-SAM docs; the default one errors out) |
| `assets/smpl/` | put `SMPL_NEUTRAL.pkl` here (licensed, not in the repo) |
| `tools/run_trial.py` | Unattended trial (bridge + SONIC + SMPL file or live publisher, optional viewer) with a fall verdict |
| `tools/stream_smpl_file.py` | Stream an SMPL file in Fast SAM's wire format (test without the camera) |
| `tools/mujoco_reference.py` | Same test against SONIC's official MuJoCo sim |
| `tools/test_hold_lowcmd.py` | Bridge self-test without SONIC |

External code is looked up under `$THIRD_PARTY` (default `~/third_party`); `tools/run_trial.py` finds Isaac Lab by
walking up to `isaaclab.sh` (override with `$ISAACLAB_PATH`).

## What had to match SONIC's training setup (A/B against the official MuJoCo sim)

| Setting | Result |
|---|---|
| PD in PhysX joint drives (`--pd implicit`), as in SONIC's Isaac Lab training | explicit per-step torque PD oscillated and fell |
| Contact material friction 1.0/1.0, combine multiply (also sim default) | Isaac's 0.5 default: walking stream fell 2/2; matched: 4/4 walked 2-4.6 m |
| Soft landing from the gantry | dropping from 0.2 m was hit-or-miss; soft landing stood 4/4 |
| Physics in a headless process + separate viewer | with rendering in the loop (~7 ms stalls) the walk fell 3/3 |
| 200 Hz physics (training rate) | keeps real time with the viewer; 500 Hz also works headless |

## Testing without the camera
```bash
cd ~/IsaacLab
python 2026_10_GIST_AILAB_Challenge/tools/run_trial.py --tag stand --seconds 60
python 2026_10_GIST_AILAB_Challenge/tools/run_trial.py --tag walk --viewer --stream walk --hold_first 20 --seconds 90
```
