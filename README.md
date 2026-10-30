# 2026_10_GIST_AILAB_Challenge

RealSense D555 카메라 한 대로 **Isaac Sim 속 Unitree G1 (Dex3 손)** 을 원격조작합니다.
사람이 걸으면 로봇이 걷고, 상체(팔·허리)와 손 그랩은 사람 동작을 따라갑니다.

```
D555 → Fast SAM 3D Body (사람 자세·위치·손) → teleop_planner.py → SONIC (NVIDIA 전신 제어 정책) → Isaac Sim G1
```

## 1. 설치

### 1-1. 이 저장소 받기

이 저장소는 **Isaac Lab 폴더 바로 안에** clone 합니다 (Isaac Lab v2.3.2, conda 환경 `env_isaaclab` 에서 확인).

```bash
cd ~/IsaacLab
git clone https://github.com/JihunHa-sh/2026_10_GIST_AILAB_Challenge.git
```
```
~/IsaacLab/
├── isaaclab.sh
└── 2026_10_GIST_AILAB_Challenge/   ← 이 저장소 (연결 코드만 들어 있음)
```

### 1-2. 외부 구성요소 설치 (새 PC 에서 한 번)

이 저장소만으로는 동작하지 않습니다. 아래 구성요소를 `~/third_party` 에 설치해야 합니다.
**사람이 직접 해야 하는 것은 두 가지뿐**입니다: HuggingFace 접근 승인(④-3), SMPL 사이트 가입(⑥).
나머지는 아래 명령을 위에서부터 그대로 복사해 실행하면 됩니다 (sudo 불필요). 각 단계 끝의 "확인" 이 통과하면 다음으로 넘어가세요.

| # | 무엇 | 왜 필요한가 |
|---|---|---|
| ① | SONIC (GR00T-WholeBodyControl) + 정책 모델 | 로봇 전신 제어 정책 |
| ② | SONIC C++ 제어기 빌드 (ZeroMQ 등 라이브러리, TensorRT 10.13, onnxruntime 1.16.3) | 정책 실행 프로그램 |
| ③ | `env_isaaclab` 에 통신 패키지 (cyclonedds, unitree_sdk2py, msgpack 등) | Isaac Sim ↔ SONIC 통신 |
| ④ | Fast SAM 3D Body + conda 환경 `fast_sam_3d_body` + 모델 가중치 | 카메라 영상 → 사람 자세 |
| ⑤ | MHR → SMPL 변환 파일 | 자세 형식 변환 |
| ⑥ | SMPL 모델 `SMPL_NEUTRAL.pkl` | 자세 형식 변환 (라이선스 때문에 직접 받기) |

**⓪ 준비물** — Ubuntu (22.04/24.04), NVIDIA 드라이버, **CUDA 12.x 툴킷** (`nvcc`), Isaac Lab v2.3.2 + conda 환경
`env_isaaclab`, `git`, `git-lfs`, `cmake` (≥3.22), `gcc/g++`, `curl`, `unzip`, RealSense D555 (USB 3 포트).
```bash
nvcc --version && git lfs version && cmake --version          # 셋 다 버전이 나오면 OK
export THIRD_PARTY=~/third_party && mkdir -p $THIRD_PARTY
export CUDA_DIR=/usr/local/cuda-12.2                            # ← 본인 PC 의 CUDA 12.x 경로로 바꾸기 (ls /usr/local)
```

**① SONIC + 정책 모델** (공개, 로그인 불필요)
```bash
cd $THIRD_PARTY && git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git && cd GR00T-WholeBodyControl
git lfs pull
conda activate env_isaaclab
python download_from_hf.py --output-dir $PWD/gear_sonic_deploy     # 정책(ONNX) + 플래너
python download_from_hf.py --sample                                 # 카메라 없는 시험용 SMPL 샘플
```
확인: `ls gear_sonic_deploy/policy/release/` 에 `model_encoder.onnx`, `model_decoder.onnx` 가 있으면 OK.

**② SONIC C++ 제어기 빌드** (약 10~20분)

SONIC 의 원래 설치 스크립트는 sudo/apt 가 필요해서, 필요한 라이브러리를 `$THIRD_PARTY/sonic_deps` 에 직접 빌드합니다.
```bash
P=$THIRD_PARTY/sonic_deps; S=$THIRD_PARTY/sonic_deps_src; mkdir -p $P $S && cd $S

# ZeroMQ (통신)
git clone -q --depth 1 -b v4.3.5 https://github.com/zeromq/libzmq.git
cmake -S libzmq -B libzmq/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$P -DBUILD_TESTS=OFF -DWITH_PERF_TOOL=OFF
cmake --build libzmq/build -j$(nproc) && cmake --install libzmq/build
git clone -q --depth 1 -b v4.10.0 https://github.com/zeromq/cppzmq.git
cmake -S cppzmq -B cppzmq/build -DCMAKE_INSTALL_PREFIX=$P -DCMAKE_PREFIX_PATH=$P -DCPPZMQ_BUILD_TESTS=OFF && cmake --install cppzmq/build

# googletest, nlohmann-json, msgpack (헤더/빌드 의존성)
git clone -q --depth 1 -b v1.14.0 https://github.com/google/googletest.git
cmake -S googletest -B googletest/build -DCMAKE_INSTALL_PREFIX=$P && cmake --build googletest/build -j$(nproc) && cmake --install googletest/build
git clone -q --depth 1 -b v3.11.3 https://github.com/nlohmann/json.git
cmake -S json -B json/build -DCMAKE_INSTALL_PREFIX=$P -DJSON_BuildTests=OFF && cmake --install json/build
git clone -q --depth 1 -b cpp-3.3.0 https://github.com/msgpack/msgpack-c.git && cp -r msgpack-c/include/* $P/include/

# Eigen (헤더만, pip 패키지에서 꺼내기)
pip download --no-deps "cmeel-eigen==3.4.0.2" -d eigen_whl && cd eigen_whl && unzip -q -o cmeel_eigen-*.whl \
  && cp -r cmeel.prefix/include/eigen3 $P/include/ && mkdir -p $P/share && cp -r cmeel.prefix/share/eigen3 $P/share/ && cd ..

# onnxruntime 1.16.3, TensorRT 10.13 (SONIC 은 정확히 10.13 필요, 로그인 없이 받아짐)
cd $THIRD_PARTY
curl -L -o ort.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz && tar -xzf ort.tgz
curl -L -o trt.tgz https://developer.download.nvidia.com/compute/machine-learning/tensorrt/10.13.3/tars/TensorRT-10.13.3.9.Linux.x86_64-gnu.cuda-12.9.tar.gz && tar -xzf trt.tgz

# SONIC 제어기 빌드
cd $THIRD_PARTY/GR00T-WholeBodyControl/gear_sonic_deploy && mkdir -p build && cd build
TensorRT_ROOT=$THIRD_PARTY/TensorRT-10.13.3.9 onnxruntime_ROOT=$THIRD_PARTY/onnxruntime-linux-x64-1.16.3 \
CUDA_HOME=$CUDA_DIR cmake -S .. -B . -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$P \
  -DZMQ_INCLUDE_DIR=$P/include -DZMQ_LIBRARY=$P/lib/libzmq.so
cmake --build . -j$(nproc)
```
확인: `ls $THIRD_PARTY/GR00T-WholeBodyControl/gear_sonic_deploy/target/release/g1_deploy_onnx_ref` 가 있으면 OK.
`run_sonic.py` 가 위 라이브러리 경로와 CUDA 경로(`$CUDA_DIR` → `/usr/local/cuda-12.2` → `/usr/local/cuda` 순서)를
자동으로 잡습니다. 첫 실행 때 TensorRT 엔진을 만드느라 1분쯤 걸립니다.

**③ Isaac Lab 환경에 통신 패키지**

Python 3.11 용 cyclonedds 0.10.2 배포판이 없어서, unitree_sdk2 에 들어 있는 DDS 라이브러리로 빌드합니다.
```bash
conda activate env_isaaclab
T=$THIRD_PARTY/GR00T-WholeBodyControl/gear_sonic_deploy/thirdparty/unitree_sdk2/thirdparty
H=$THIRD_PARTY/cyclonedds_home && mkdir -p $H/bin && ln -sfn $T/include $H/include && ln -sfn $T/lib/x86_64 $H/lib
CYCLONEDDS_HOME=$H pip install "cyclonedds==0.10.2"
pip install --no-deps -e $THIRD_PARTY/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python
pip install joblib pyzmq msgpack
```
확인: `python -c "import cyclonedds, unitree_sdk2py, msgpack; print('ok')"` 가 `ok` 를 출력하면 OK.

**④ Fast SAM 3D Body** (약 20~30분)

④-1. 저장소와 conda 환경 `fast_sam_3d_body`
```bash
cd $THIRD_PARTY && git clone https://github.com/yangtiming/Fast-SAM-3D-Body.git && cd Fast-SAM-3D-Body
bash setup_env.sh            # 환경 생성. 마지막에 detectron2 설치가 실패해도 다음 단계에서 해결됨
```

④-2. 환경 수정 (이 PC 에서 필요했던 것들을 그대로)
```bash
conda activate fast_sam_3d_body
E=$(conda info --base)/envs/fast_sam_3d_body; NVJ=$E/lib/python3.11/site-packages/nvidia/nvjitlink/lib
# 시스템 CUDA 경로가 섞이면 torch 가 깨지므로 이 환경의 라이브러리 경로를 고정
conda env config vars set -n fast_sam_3d_body LD_LIBRARY_PATH="$NVJ"
conda deactivate && conda activate fast_sam_3d_body
export CUDA_HOME=$E PATH=$E/bin:$PATH
pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps
pip install ultralytics git+https://github.com/microsoft/MoGe.git tensorrt-cu12 onnx onnxruntime-gpu nvtx \
            smplx numpy scipy opencv-python tqdm pyzmq pyrealsense2 && pip install chumpy --no-build-isolation
# 위 설치가 torch 버전을 올려버리므로 detectron2 와 맞는 버전으로 되돌림
pip install --no-deps --force-reinstall torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install --no-deps triton==3.1.0
```
확인: `python -c "import torch, detectron2, pyrealsense2; print(torch.__version__, torch.cuda.is_available())"` →
`2.5.1+cu124 True` 이면 OK.

④-3. 모델 가중치 — **[사람이 할 일]** https://huggingface.co/facebook/sam-3d-body-dinov3 에서 로그인 후
접근 신청 (승인까지 시간이 걸림). 승인되면 https://huggingface.co/settings/tokens 에서 **Read** 토큰을 만들고:
```bash
cd $THIRD_PARTY/Fast-SAM-3D-Body && conda activate fast_sam_3d_body
hf auth login                # 토큰 붙여넣기 (터미널에만 입력, 다른 곳에 공유하지 말 것)
python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/sam-3d-body-dinov3', local_dir='checkpoints/sam-3d-body-dinov3')"
mkdir -p checkpoints/yolo && curl -L -o checkpoints/yolo/yolo11m-pose.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-pose.pt
```
확인: `ls checkpoints/sam-3d-body-dinov3/model.ckpt checkpoints/yolo/yolo11m-pose.pt` 둘 다 있으면 OK.
(403 오류 = 아직 승인 전이거나, 토큰이 Read 가 아닌 세분화(fine-grained) 토큰인 경우)

**⑤ MHR → SMPL 변환 파일** (공개)
```bash
cd $THIRD_PARTY && git clone --depth 1 --filter=blob:none --sparse https://github.com/facebookresearch/MHR.git mhr
cd mhr && git sparse-checkout set tools/mhr_smpl_conversion/assets && git lfs pull --include "tools/mhr_smpl_conversion/assets/*"
```
확인: `ls tools/mhr_smpl_conversion/assets/` 에 `mhr2smpl_mapping.npz`, `mhr_face_mask.ply` 가 있으면 OK.

**⑥ SMPL 모델** — **[사람이 할 일]** https://smplify.is.tue.mpg.de/ 가입 → Downloads 에서
`mpips_smplify_public_v2.zip` 를 받아 `~/Downloads` 에 두고:
```bash
unzip -p ~/Downloads/mpips_smplify_public_v2.zip smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl \
  > ~/IsaacLab/2026_10_GIST_AILAB_Challenge/assets/smpl/SMPL_NEUTRAL.pkl
```
확인: `ls -la ~/IsaacLab/2026_10_GIST_AILAB_Challenge/assets/smpl/SMPL_NEUTRAL.pkl` 크기가 수십 MB 면 OK.

**⑦ 전체 설치 확인**
```bash
# 시뮬 + SONIC (카메라 없이, 약 1분) → 마지막 줄에 STOOD 가 나오면 성공
cd ~/IsaacLab && conda activate env_isaaclab
python 2026_10_GIST_AILAB_Challenge/tools/run_trial.py --tag check --seconds 50

# 카메라 + 자세 추정 → 카메라 앞에 혼자 서서 `Live: ... publish=50.0Hz` 가 나오면 성공 (Ctrl+C 로 종료)
conda activate fast_sam_3d_body
python 2026_10_GIST_AILAB_Challenge/run_fastsam_publisher.py
```
터미널 4의 첫 실행은 HuggingFace 에서 공개 모델(MoGe)을 자동으로 받으므로 인터넷이 필요합니다 (이후엔 캐시 사용).
카메라를 못 찾는다는 오류가 나면 RealSense 가 USB 3 포트에 꽂혔는지 확인하고, 그래도 안 되면 librealsense 의
udev 규칙을 설치하세요 (관리자 권한 필요).
외부 구성요소를 `~/third_party` 가 아닌 곳에 설치했다면 모든 터미널에서 `export THIRD_PARTY=<경로>` 를 먼저 하세요.

## 2. 터미널 명령 (5개, 이 순서대로)

모든 터미널에서 먼저 `cd ~/IsaacLab` 하세요.

| 터미널 | conda 환경 | 명령 | 역할 |
|---|---|---|---|
| 1 | `env_isaaclab` | `nice -n 10 ./isaaclab.sh -p 2026_10_GIST_AILAB_Challenge/isaac_viewer.py` | Isaac Sim 화면 (창이 뜰 때까지 기다리기) |
| 2 | `env_isaaclab` | `./isaaclab.sh -p 2026_10_GIST_AILAB_Challenge/isaac_sonic_bridge.py --headless` | 물리 시뮬레이션 |
| 3 | `env_isaaclab` | `python 2026_10_GIST_AILAB_Challenge/run_sonic.py --input-type zmq_manager --zmq-port 5559` | SONIC 제어기 |
| 4 | `fast_sam_3d_body` | `python 2026_10_GIST_AILAB_Challenge/run_fastsam_publisher.py` | 카메라 + 자세 추정 |
| 5 | `env_isaaclab` | `python 2026_10_GIST_AILAB_Challenge/teleop_planner.py` | 원격조작 (키 입력은 여기) |

터미널 4는 시작할 때 카메라 IMU로 중력 방향을 잡으니 **카메라를 움직이지 마세요.**

## 3. 사용법

1. 카메라에서 2~3 m 떨어져 **카메라를 정면으로 보고 차렷 자세**(팔을 쭉 펴서 내림)로 섭니다.
   화면에 **나 혼자** 머리부터 발까지 보여야 합니다.
2. **터미널 5에서 `ENTER`** → 로봇 제어 시작 + 팔/방향 보정 (지금 내 차렷 자세 = 로봇 차렷 자세).
3. **Isaac Sim 창(터미널 1)에서 `9`** → 로봇이 바닥에 내려와 섭니다.
4. 움직이면 로봇이 따라옵니다.
   - **걷기**: 앞·뒤·옆으로 걸으면 로봇도 같은 방향으로 걷습니다. 몸을 돌리면 로봇도 돕니다.
   - **팔·허리**: 양손목 위치/방향과 상체를 따라갑니다.
   - **손**: 주먹을 쥐면 로봇 손이 꽉 쥐고(그랩), 펴면 완전히 폅니다.

터미널 5 키 (입력 후 `ENTER`):

| 키 | 동작 |
|---|---|
| `ENTER` 만 | 다시 보정 (팔이 어긋나 보이면 차렷 자세로 서서 누르기) |
| `l` | 걷기 끄기/켜기 (끄면 제자리에서 상체만 따라감) |
| `o` | 로봇 정지 |

Isaac Sim 창 키: `9` 로봇 내리기/다시 잡기, `R` 처음 위치로 리셋, `7`/`8` 매다는 높이 조절.

로그 보는 법:
- 터미널 5 `[loco]` : `face-to-cam` 이 카메라를 볼 때 0 근처, `cmd` 가 서 있을 때 `IDLE` 이면 정상.
- 터미널 4 `[hands]` : 손 굽힘 값과 `GRAB`/`open` 상태. `Live: ... publish=50.0Hz` 면 정상 송신 중.
- 터미널 2 : `rtf=1.00` 이 유지돼야 합니다 (낮아지면 다른 무거운 프로그램을 끄세요).

## 4. 자주 쓰는 옵션

| 대상 | 옵션 | 설명 |
|---|---|---|
| 터미널 5 | `--walk-start 0.2 --walk-stop 0.1` | 걷기 시작/정지 속도 기준 [m/s] |
| 터미널 5 | `--speed-gain 1.0 --max-speed 0.6` | 로봇 속도 = 사람 속도 × gain (최대값) |
| 터미널 5 | `--turn-deadband 10` | 이 각도[deg]보다 작은 몸 회전은 무시 |
| 터미널 4 | `--grab-close 0.5 --grab-open 0.3` | 손을 쥐는/펴는 기준 (0 = 편 손, 1 = 주먹) |
| 터미널 4 | `--hand-mode fingers` | 그랩 대신 손가락별로 비례해서 굽힘 |

**전신 따라하기 모드** (걷기 없이 다리까지 사람 자세를 그대로 따라함): 터미널 3을
`python 2026_10_GIST_AILAB_Challenge/run_sonic.py --input-type zmq` 로 실행하고 터미널 5는 쓰지 않습니다.
터미널 3에서 `]` → Isaac Sim 창에서 `9` → 차렷 자세로 서서 터미널 3에서 `ENTER`.

## 5. 파일 구성

| 파일 | 내용 |
|---|---|
| `isaac_viewer.py` | Isaac Sim 화면 (물리 상태를 받아 그리기만 함, 키 입력 전달) |
| `isaac_sonic_bridge.py` | Isaac Sim 물리 (SONIC 의 MuJoCo 시뮬 대신, DDS 로 실제 G1 처럼 동작) |
| `run_sonic.py` | SONIC C++ 제어기 실행 |
| `run_fastsam_publisher.py` | D555 + Fast SAM 3D Body (자세·골반 위치·손 → ZMQ :5556) |
| `teleop_planner.py` | 원격조작: 골반 위치 → 걷기 명령, 자세 → 상체 3점(손목·목), 손 그랩 → SONIC (ZMQ :5559) |
| `vr3pt.py` | SMPL 자세 → SONIC 의 VR 3점 형식 (PICO 원격조작 코드 재사용) |
| `hand_retarget.py` | 손 키포인트 → 손가락 굽힘 → Dex3 관절값 |
| `g1_dex3.py` | SONIC 학습용 G1 모델의 Dex3 손가락 관절을 움직이게 설정 |
| `config/obs_config_fastsam.yaml` | SONIC 관측 설정 |
| `tools/` | 카메라 없이 하는 시험 (`run_trial.py`, `planner_test.py`, `stream_smpl_file.py` 등) |

카메라 없이 시뮬만 확인:
```bash
python 2026_10_GIST_AILAB_Challenge/tools/run_trial.py --tag check --seconds 50                       # 서 있기 → STOOD
python 2026_10_GIST_AILAB_Challenge/tools/run_trial.py --tag arms --viewer --planner --planner_args "--arms --start-delay 25" --release_after 3 --seconds 85
```

## 참고: Isaac Sim 에서 SONIC 을 안정적으로 돌리기 위해 맞춘 것

| 설정 | 이유 |
|---|---|
| PD 제어를 PhysX 관절 드라이브에 넣음 (SONIC 학습과 동일) | 매 스텝 토크 계산 방식은 진동하다 넘어짐 |
| 바닥 마찰 1.0 (학습과 동일) | Isaac 기본값 0.5 에서는 걸을 때 발이 미끄러짐 |
| 물리는 화면 없는 프로세스, 화면은 별도 뷰어 | 물리 루프 안에서 렌더링하면 제어 타이밍이 흔들려 떨리고 넘어짐 |
| 200 Hz 물리 (학습과 동일), 매달린 상태에서 천천히 내려놓기 | 실시간 유지, 안정적인 착지 |
| Dex3 손가락은 같은 손 안끼리 충돌 끔 | 충돌 모양이 겹쳐 엄지가 밀려남 (물체와의 충돌은 유지) |
