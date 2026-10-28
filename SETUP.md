# Setup (first time)

Tested on Ubuntu 24.04, RTX 4090 (driver 580), Isaac Lab v2.3.2 in the conda env `env_isaaclab`, RealSense D555.
Everything below installs into user space (no sudo). External code goes to `$THIRD_PARTY` (default `~/third_party`).

```bash
export THIRD_PARTY=~/third_party && mkdir -p $THIRD_PARTY
cd ~/IsaacLab && git clone https://github.com/JihunHa-sh/2026_10_GIST_AILAB_Challenge.git
REPO=~/IsaacLab/2026_10_GIST_AILAB_Challenge
```

## 1. SONIC (GR00T-WholeBodyControl) + models

```bash
cd $THIRD_PARTY && git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git && cd GR00T-WholeBodyControl
git lfs pull
conda activate env_isaaclab
python download_from_hf.py --output-dir $PWD/gear_sonic_deploy     # ONNX policy + planner (public)
python download_from_hf.py --sample                                 # optional: SMPL sample for tools/run_trial.py --stream walk
bash install_scripts/install_mujoco_sim.sh                          # optional: official MuJoCo sim (tools/mujoco_reference.py)
```

## 2. Build the SONIC C++ deployment without sudo

SONIC's `scripts/install_deps.sh` needs sudo/apt. Instead build its libraries into `$THIRD_PARTY/sonic_deps`:

```bash
P=$THIRD_PARTY/sonic_deps; S=$THIRD_PARTY/sonic_deps_src; mkdir -p $P $S && cd $S
git clone -q --depth 1 -b v4.3.5 https://github.com/zeromq/libzmq.git
cmake -S libzmq -B libzmq/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$P -DBUILD_TESTS=OFF -DWITH_PERF_TOOL=OFF
cmake --build libzmq/build -j16 && cmake --install libzmq/build
git clone -q --depth 1 -b v4.10.0 https://github.com/zeromq/cppzmq.git
cmake -S cppzmq -B cppzmq/build -DCMAKE_INSTALL_PREFIX=$P -DCMAKE_PREFIX_PATH=$P -DCPPZMQ_BUILD_TESTS=OFF && cmake --install cppzmq/build
git clone -q --depth 1 -b v1.14.0 https://github.com/google/googletest.git
cmake -S googletest -B googletest/build -DCMAKE_INSTALL_PREFIX=$P && cmake --build googletest/build -j16 && cmake --install googletest/build
git clone -q --depth 1 -b v3.11.3 https://github.com/nlohmann/json.git
cmake -S json -B json/build -DCMAKE_INSTALL_PREFIX=$P -DJSON_BuildTests=OFF && cmake --install json/build
git clone -q --depth 1 -b cpp-3.3.0 https://github.com/msgpack/msgpack-c.git && cp -r msgpack-c/include/* $P/include/   # header-only, no boost
# Eigen (gitlab.com was unreachable here): headers + cmake config from the cmeel wheel
pip download --no-deps "cmeel-eigen==3.4.0.2" -d eigen_whl && cd eigen_whl && unzip -q -o cmeel_eigen-*.whl \
  && cp -r cmeel.prefix/include/eigen3 $P/include/ && mkdir -p $P/share && cp -r cmeel.prefix/share/eigen3 $P/share/ && cd ..

# onnxruntime 1.16.3 and TensorRT 10.13 (SONIC requires exactly 10.13; the tarball needs no login)
cd $THIRD_PARTY
curl -L -o ort.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz && tar -xzf ort.tgz
curl -L -o trt.tgz https://developer.download.nvidia.com/compute/machine-learning/tensorrt/10.13.3/tars/TensorRT-10.13.3.9.Linux.x86_64-gnu.cuda-12.9.tar.gz && tar -xzf trt.tgz

# build (CUDA toolkit 12.x; here /usr/local/cuda-12.2)
cd $THIRD_PARTY/GR00T-WholeBodyControl/gear_sonic_deploy && mkdir -p build && cd build
TensorRT_ROOT=$THIRD_PARTY/TensorRT-10.13.3.9 onnxruntime_ROOT=$THIRD_PARTY/onnxruntime-linux-x64-1.16.3 \
CUDA_HOME=/usr/local/cuda-12.2 cmake -S .. -B . -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$P \
  -DZMQ_INCLUDE_DIR=$P/include -DZMQ_LIBRARY=$P/lib/libzmq.so
cmake --build . -j16          # -> gear_sonic_deploy/target/release/g1_deploy_onnx_ref
```
`run_sonic.py` sets `LD_LIBRARY_PATH` for these libraries. The first run builds TensorRT engines (~1 min, cached).

## 3. Unitree DDS in the Isaac Lab env

PyPI has no cyclonedds 0.10.2 wheel for Python 3.11; build it against the ddsc bundled with unitree_sdk2:
```bash
conda activate env_isaaclab
T=$THIRD_PARTY/GR00T-WholeBodyControl/gear_sonic_deploy/thirdparty/unitree_sdk2/thirdparty
H=$THIRD_PARTY/cyclonedds_home && mkdir -p $H/bin && ln -sfn $T/include $H/include && ln -sfn $T/lib/x86_64 $H/lib
CYCLONEDDS_HOME=$H pip install "cyclonedds==0.10.2"
pip install --no-deps -e $THIRD_PARTY/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python
pip install joblib pyzmq
```

## 4. Fast SAM 3D Body

```bash
cd $THIRD_PARTY && git clone https://github.com/yangtiming/Fast-SAM-3D-Body.git && cd Fast-SAM-3D-Body
bash setup_env.sh     # creates conda env fast_sam_3d_body; detectron2 may fail on the first try, see below
```
Fixes that were needed here:
```bash
E=~/miniconda3/envs/fast_sam_3d_body; NVJ=$E/lib/python3.11/site-packages/nvidia/nvjitlink/lib
# a shell-wide LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64 makes torch cu124 fail (nvJitLink); pin it for this env
conda env config vars set -n fast_sam_3d_body LD_LIBRARY_PATH="$NVJ"
export LD_LIBRARY_PATH="$NVJ" CUDA_HOME=$E PATH=$E/bin:$PATH
pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps
pip install ultralytics git+https://github.com/microsoft/MoGe.git tensorrt-cu12 onnx onnxruntime-gpu nvtx \
            smplx numpy scipy opencv-python tqdm pyzmq pyrealsense2 && pip install chumpy --no-build-isolation
# ultralytics/MoGe upgrade torch -> put back the version detectron2 was built against
pip install --no-deps --force-reinstall torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install --no-deps triton==3.1.0
```
Weights:
```bash
# SAM 3D Body is gated: request access at https://huggingface.co/facebook/sam-3d-body-dinov3 (manual approval),
# then `hf auth login` with a *Read* token (a fine-grained token without public-gated-repo access gets 403)
python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/sam-3d-body-dinov3', local_dir='checkpoints/sam-3d-body-dinov3')"
mkdir -p checkpoints/yolo && curl -L -o checkpoints/yolo/yolo11m-pose.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-pose.pt
```

## 5. MHR → SMPL conversion assets
```bash
cd $THIRD_PARTY && git clone --depth 1 --filter=blob:none --sparse https://github.com/facebookresearch/MHR.git mhr
cd mhr && git sparse-checkout set tools/mhr_smpl_conversion/assets && git lfs pull --include "tools/mhr_smpl_conversion/assets/*"
```

## 6. SMPL model (licensed, not redistributable)
Register at https://smplify.is.tue.mpg.de/, download `mpips_smplify_public_v2.zip`, then:
```bash
unzip -p mpips_smplify_public_v2.zip smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl \
  > $REPO/assets/smpl/SMPL_NEUTRAL.pkl
```

## 7. Check
```bash
cd ~/IsaacLab && conda activate env_isaaclab
python 2026_10_GIST_AILAB_Challenge/tools/run_trial.py --tag check --seconds 50    # expect: STOOD
```
