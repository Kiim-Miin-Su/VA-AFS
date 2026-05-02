# VA-AFS Demo README

> **목표**  
> 직접 촬영한 영상에서 MediaPipe로 skeleton coordinate sequence를 추출하고, VA-AFS threshold gate를 적용해 중요한 skeleton frame만 선택한다.

---

## 0. Related Publications & Terms, Conditions of Use

| Terms & Conditions of Use

The datasets are released for academic research only, and are free to researchers from educational or research institutes for non-commercial purposes.

The use of these two datasets is governed by the following terms and conditions:
• Without the expressed permission of the ROSE Lab, any of the following will be considered illegal: redistribution, derivation or generation of a new dataset from this dataset, and commercial usage of any of these datasets in any way or form, either partially or in its entirety.
• For the sake of privacy, images of all subjects in any of these datasets are only allowed for the demonstration in academic publications and presentations.
• All users of "NTU RGB+D" and "NTU RGB+D 120" action recognition datasets agree to indemnify, defend and hold harmless, the ROSE Lab and its officers, employees, and agents, individually and collectively, from any and all losses, expenses, and damages.

If interested, researchers can register for an account, submit the request form and accept the Release Agreement. We will validate your request and grant approval for downloading the datasets.The LoginID can be used for both "NTU RGB+D" and "NTU RGB+D 120".

All publications using "NTU RGB+D" or "NTU RGB+D 120" Action Recognition Database or any of the derived datasets(see Section 8) should include the following acknowledgement: "(Portions of) the research in this paper used the NTU RGB+D (or NTU RGB+D 120) Action Recognition Dataset made available by the ROSE Lab at the Nanyang Technological University, Singapore."

---

| Related Publications

Furthermore, these publications should cite the following papers:

    Amir Shahroudy, Jun Liu, Tian-Tsong Ng, Gang Wang, "NTU RGB+D: A Large Scale Dataset for 3D Human Activity Analysis", IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016 [PDF].
    Jun Liu, Amir Shahroudy, Mauricio Perez, Gang Wang, Ling-Yu Duan, Alex C. Kot, "NTU RGB+D 120: A Large-Scale Benchmark for 3D Human Activity Understanding", IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2019. [PDF].

Some related works on RGB+D action recognition:

    Amir Shahroudy, Tian-Tsong Ng, Qingxiong Yang, Gang Wang, "Multimodal Multipart Learning for Action Recognition in Depth Videos", TPAMI, 2016.
    Amir Shahroudy, Tian-Tsong Ng, Yihong Gong, Gang Wang, "Deep Multimodal Feature Analysis for Action Recognition in RGB+D Videos" TPAMI, 2018.
    Amir Shahroudy, Gang Wang, Tian Tsong Ng, "Multi-modal Feature Fusion for Action Recognition in RGB-D Sequences", ISCCSP, 2014.
    Jun Liu, Amir Shahroudy, Dong Xu, Gang Wang, "Spatio-Temporal LSTM with Trust Gates for 3D Human Action Recognition", ECCV, 2016.
    Jun Liu, Gang Wang, Ping Hu, Ling-Yu Duan, Alex C. Kot, "Global Context-Aware Attention LSTM Networks for 3D Action Recognition", CVPR, 2017.
    Jun Liu, Amir Shahroudy, Dong Xu, Alex C. Kot, Gang Wang, "Skeleton-Based Action Recognition Using Spatio-Temporal LSTM Network with Trust Gates", TPAMI, 2018.
    Jun Liu, Gang Wang, Ling-Yu Duan, Kamila Abdiyeva, Alex C. Kot, "Skeleton-Based Human Action Recognition with Global Context-aware Attention LSTM Networks", TIP, 2018.
    Jun Liu, Amir Shahroudy, Gang Wang, Ling-Yu Duan, Alex C. Kot, "Skeleton-Based Online Action Prediction Using Scale Selection Network", TPAMI, 2019.
    Siyuan Yang, Jun Liu, Shijian Lu, Er Meng Hwa, and Alex Kot, "Collaborative Learning of Gesture Recognition and 3D Hand Pose Estimation with Multi-Order Feature Analysis", ECCV 2020.
    Siyuan Yang, Jun Liu, Shijian Lu, Er Meng Hwa, and Alex Kot, "Skeleton Cloud Colorization for Unsupervised 3D Action Representation Learning", ICCV 2021.

## 1. 전체 흐름

```text
Video File
↓
MediaPipe Pose
↓
Skeleton Coordinate Sequence
skeleton.shape = (T, V, C)
↓
VA-AFS Online Threshold Sampling
↓
Selected Skeleton Frames + Selected Frame Images
↓
Online Score / Raw Beta / Motion Change Plot
```

현재 단계의 목표는 **BlockGCN 학습 전**, VA-AFS가 실제 skeleton sequence에서 중요한 frame을 online 방식으로 잘 고르는지 확인하는 것이다.

---

## 2. 폴더 구조

```text
VA-AFS/
├── videos/
│   └── squat_01.mp4
├── outputs/
│   ├── skeleton/
│   │   ├── npy/
│   │   │   └── squat_01_skeleton.npy
│   │   └── csv/
│   │       └── squat_01_skeleton.csv
│   ├── previews/
│   │   └── mp4/
│   │       └── squat_01_preview.mp4
│   └── threshold/
│       ├── npy/
│       │   └── squat_01_skeleton_tau0.9_k13_sampled.npy
│       ├── plots/
│       │   └── squat_01_skeleton_tau0.9_k13_selection.png
│       └── selected_frames/
│           └── squat_01_skeleton_tau0.9_k13/
│               ├── squat_01_frame_000000.jpg
│               └── squat_01_frame_000013.jpg
├── extract_skeleton_mediapipe.py
├── va_afs_threshold.py
├── apply_va_afs_to_blockgcn_npz.py
├── run_blockgcn_acc.py
├── constants.py
└── README.md
```

출력은 기능과 확장자 기준으로 나뉜다.

| 경로 | 의미 |
| ---- | ---- |
| `outputs/skeleton/npy/` | MediaPipe skeleton `.npy` |
| `outputs/skeleton/csv/` | frame/joint 좌표 CSV |
| `outputs/previews/mp4/` | pose landmark preview video |
| `outputs/threshold/npy/` | VA-AFS가 선택한 skeleton sequence |
| `outputs/threshold/plots/` | 선택 과정 시각화 plot |
| `outputs/threshold/selected_frames/` | 실제 선택된 원본 frame 이미지 |
| `outputs/blockgcn_npz/` | BlockGCN 평가용 VA-AFS 적용 NTU `.npz` |
| `outputs/blockgcn_configs/` | BlockGCN 실행용 임시 config |
| `outputs/blockgcn_acc/` | BlockGCN accuracy 로그와 score |

---

## 3. 가상환경 생성

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows CMD:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

가상환경이 켜지면 터미널 앞에 `(.venv)`가 표시된다.

---

## 4. 의존성 설치

### requirements.txt

```txt
opencv-python
mediapipe
numpy
pandas
matplotlib
tqdm
```

설치:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt`가 아직 없다면 직접 설치한다.

```bash
python -m pip install opencv-python mediapipe numpy pandas matplotlib tqdm
```

설치 확인:

```bash
python -c "import cv2, mediapipe, numpy, pandas, matplotlib, tqdm; print('ok')"
```

---

## 5. 영상 준비

`videos/` 폴더에 테스트 영상을 넣는다.

```text
videos/squat_01.mp4
```

촬영 기준:

```text
1. 한 사람만 나오게 촬영
2. 전신이 화면 안에 들어오게 촬영
3. 카메라는 고정
4. 5~10초 정도 촬영
5. 배경은 단순하게
6. 30 FPS 권장
```

처음 테스트하기 좋은 동작:

```text
stand_still
squat
arm_raise
walking
jump
```

---

## 6. Step 1: MediaPipe로 skeleton 추출

현재 위치가 `VA-AFS/` 폴더라면:

```bash
python extract_skeleton_mediapipe.py --video videos/squat_01.mp4
```

다른 위치에서 실행한다면 전체 경로를 사용한다.

```bash
python /path/to/VA-AFS/extract_skeleton_mediapipe.py \
  --video /path/to/VA-AFS/videos/squat_01.mp4
```

출력 파일:

```text
outputs/skeleton/npy/squat_01_skeleton.npy
outputs/skeleton/csv/squat_01_skeleton.csv
outputs/previews/mp4/squat_01_preview.mp4
```

| 파일 | 의미 |
| ---- | ---- |
| `*_skeleton.npy` | VA-AFS에 사용할 skeleton numpy 파일 |
| `*_skeleton.csv` | frame, joint, x, y, visibility를 저장한 CSV |
| `*_preview.mp4` | MediaPipe pose landmark가 그려진 확인용 영상 |

---

## 7. 추출된 skeleton shape

MediaPipe Pose 기준:

```python
skeleton.shape = (T, 33, 3)
```

| 축 | 의미 |
| -- | ---- |
| `T` | 영상 frame 수 |
| `33` | MediaPipe Pose landmark 개수 |
| `3` | `x`, `y`, `visibility` |

VA-AFS는 좌표만 사용한다.

```python
skeleton_xy = skeleton[:, :, :2]
```

결과:

```python
skeleton_xy.shape = (T, 33, 2)
```

---

## 8. Step 2: VA-AFS online threshold sampling 적용

기본 실행:

```bash
python va_afs_threshold.py \
  --npy outputs/skeleton/npy/squat_01_skeleton.npy
```

옵션을 직접 지정하는 예시:

```bash
python va_afs_threshold.py \
  --npy outputs/skeleton/npy/squat_01_skeleton.npy \
  --video videos/squat_01.mp4 \
  --tau 0.9 \
  --k_max 13 \
  --window_size 13
```

| 인자 | 의미 | 기본값 |
| ---- | ---- | -----: |
| `--npy` | MediaPipe로 추출한 skeleton `.npy` 경로 | 필수 |
| `--video` | 선택 frame 이미지를 저장할 원본 영상 경로 | 자동 탐색 |
| `--tau` | 0~1 online score threshold | `TAU[-1] = 0.9` |
| `--k_max` | 최대 연속 skip 허용 frame 수 | `K_MAX[-1] = 13` |
| `--window_size` | motion feature 계산에 사용할 최근 frame 수 | `WINDOW_SIZE[-1] = 13` |

`--video`를 생략하면 `.npy` 이름에서 `_skeleton`을 제거한 뒤 `videos/` 아래에서 같은 이름의 영상을 찾는다. 예를 들어 `squat_01_skeleton.npy`는 `videos/squat_01.mp4`, `videos/squat_01.avi` 등을 탐색한다.

---

## 9. window_size와 warmup 의미

`window_size`는 두 군데에서 쓰인다.

| 용도 | 설명 |
| ---- | ---- |
| Motion feature window | 현재 frame의 displacement, velocity, acceleration, window 내부 표준편차를 계산할 때 최근 최대 `window_size`개 frame을 참조한다. |
| Warmup length | online beta 평균/표준편차가 안정되기 전까지 처음 `window_size`개 frame을 강제로 선택한다. |

`window_size = 13`이면 매 frame마다 최근 최대 13개의 frame을 보고 motion feature를 계산한다.

```text
T = 60
window_size = 13
```

| 현재 frame t | 참조 window |
| -----------: | ----------- |
| 0 | 0 |
| 1 | 0~1 |
| 2 | 0~2 |
| 12 | 0~12 |
| 13 | 1~13 |
| 14 | 2~14 |
| ... | ... |
| 59 | 47~59 |

초기에는 online beta 통계가 불안정하므로 **처음 `window_size`개의 frame은 warmup으로 강제 선택**한다.

```text
t < window_size  →  g_t = 1
```

튜닝 관점:

```text
window_size 증가 → 더 긴 시간 변화량을 보지만 warmup으로 선택되는 frame도 증가
window_size 감소 → 더 민감하고 빠르게 반응하지만 beta 통계와 motion feature가 더 흔들릴 수 있음
```

BlockGCN 평가에서는 VA-AFS의 `window_size`와 BlockGCN feeder의 `window_size`를 구분해야 한다.

```text
VA-AFS window_size       : 어떤 frame을 선택할지 결정하는 online gate window
BlockGCN feeder window_size: 선택된 sequence를 모델 입력 길이로 resize하는 길이, config 기본 64
```

---

## 10. VA-AFS 핵심 수식

### Displacement

```math
d_{t,j} = p_{t,j} - p_{t-W+1,j}
```

### Velocity

```math
v_{t,j} = p_{t,j} - p_{t-1,j}
```

### Acceleration

```math
a_{t,j} = v_{t,j} - v_{t-1,j}
```

### Historical Standard Deviation inside Window

```math
\sigma_j = \text{std}(p_{t-W+1,j}, \ldots, p_{t,j})
```

### Motion Importance

```math
m_{t,j}
=
\|d_{t,j}\|
+
\lambda_v \|v_{t,j}\|
+
\lambda_a \|a_{t,j}\|
```

### Variance-Aware Joint Score

```math
r_{t,j}
=
\frac{m_{t,j}}{\sigma_j + \epsilon}
```

### Raw Frame Importance

```math
\beta_t
=
\lambda_{\text{mean}}
\cdot
\frac{1}{V}
\sum_{j=1}^{V}
r_{t,j}
+
\lambda_{\text{max}}
\cdot
\max_j r_{t,j}
```

`β_t`는 0~1 값이 아니다. 움직임이 window 내 변동성보다 크면 1을 넘을 수 있다.

### Online Normalized Score

현재 frame 판단에는 현재 frame을 포함하지 않은 이전 beta 통계만 사용한다.

```math
z_t
=
\frac{\beta_t - \mu^\beta_{t-1}}
{\sigma^\beta_{t-1} + \epsilon}
```

```math
score_t = \text{sigmoid}(z_t)
```

따라서 `score_t`는 0~1 사이의 값이다. 초기 beta 통계가 충분하지 않은 구간은 bootstrap으로 `z_t = 0`, `score_t = 0.5`를 사용한다.

### Online Frame Selection Gate

```math
g_t =
\begin{cases}
1, & t < W \\
1, & score_t \ge \tau \\
1, & t - t_{\text{last}} \ge K_{\max} \\
0, & \text{otherwise}
\end{cases}
```

의미:

```text
초기 window_size frame은 warmup으로 선택
score_t가 tau 이상이면 선택
score_t가 낮아도 K_max frame 이상 연속 skip하면 강제 선택
```

---

## 11. 출력 결과 확인

VA-AFS 실행 후 출력 예시:

```text
Original shape: (42, 33, 2)
Sampled shape: (15, 33, 2)
Processed frame ratio: 0.357
Selected indices: [ 0  1  2 ... 12 25 38] ...
Saved sampled skeleton: outputs/threshold/npy/squat_01_skeleton_tau0.9_k13_sampled.npy
Saved plot: outputs/threshold/plots/squat_01_skeleton_tau0.9_k13_selection.png
Saved selected frame images: 15 files
Selected frame image dir: outputs/threshold/selected_frames/squat_01_skeleton_tau0.9_k13
```

| 출력 | 의미 |
| ---- | ---- |
| `Original shape` | 원본 skeleton shape |
| `Sampled shape` | 선택된 skeleton shape |
| `Processed frame ratio` | 전체 frame 중 선택된 frame 비율 |
| `Selected indices` | 선택된 frame 번호 |
| `Saved selected frame images` | 실제 원본 영상에서 저장한 선택 frame 이미지 수 |

---

## 12. 결과 파일

예시:

```text
outputs/threshold/npy/squat_01_skeleton_tau0.9_k13_sampled.npy
outputs/threshold/plots/squat_01_skeleton_tau0.9_k13_selection.png
outputs/threshold/selected_frames/squat_01_skeleton_tau0.9_k13/squat_01_frame_000000.jpg
```

| 파일 | 의미 |
| ---- | ---- |
| `*_sampled.npy` | 선택된 skeleton sequence |
| `*_selection.png` | online score, raw beta, motion change, 선택 frame 시각화 |
| `*_frame_*.jpg` | 실제 선택된 원본 frame 이미지 |

`*_selection.png`에서 확인할 것:

```text
1. 상단: score_t가 tau 이상인 frame이 선택되는가?
2. 상단: warmup, tau, k_max 선택 이유가 marker로 구분되는가?
3. 중단: raw beta가 adaptive threshold를 넘는 시점이 score 선택과 맞는가?
4. 하단: displacement/velocity/acceleration 변화가 큰 구간에서 선택되는가?
5. annotation: total frame -> selected frame, 선택 비율이 적절한가?
```

---

## 13. tau / k_max / window_size 실험

`tau`는 이제 raw `β_t`가 아니라 0~1 online score에 적용된다.

```bash
python va_afs_threshold.py --npy outputs/skeleton/npy/squat_01_skeleton.npy --tau 0.5 --k_max 13 --window_size 13
python va_afs_threshold.py --npy outputs/skeleton/npy/squat_01_skeleton.npy --tau 0.7 --k_max 13 --window_size 13
python va_afs_threshold.py --npy outputs/skeleton/npy/squat_01_skeleton.npy --tau 0.9 --k_max 13 --window_size 13
```

`k_max`도 바꿔본다.

```bash
python va_afs_threshold.py --npy outputs/skeleton/npy/squat_01_skeleton.npy --tau 0.9 --k_max 5 --window_size 13
python va_afs_threshold.py --npy outputs/skeleton/npy/squat_01_skeleton.npy --tau 0.9 --k_max 8 --window_size 13
python va_afs_threshold.py --npy outputs/skeleton/npy/squat_01_skeleton.npy --tau 0.9 --k_max 13 --window_size 13
```

예상 경향:

```text
tau 증가 → score_t 기준 선택 frame 감소
k_max 감소 → 강제 선택 frame 증가
k_max 증가 → 더 많이 skip 가능
window_size 증가 → warmup 선택 frame 증가, motion feature가 더 긴 구간을 참조
```

---

## 14. 현재 단계에서 봐야 할 핵심 지표

### Processed Frame Ratio

```math
\text{Processed Frame Ratio}
=
\frac{T'}{T}
```

| 기호 | 의미 |
| ---- | ---- |
| \(T\) | 원본 frame 수 |
| \(T'\) | 선택된 frame 수 |

### Frame Reduction

```math
\text{Frame Reduction}
=
1 - \frac{T'}{T}
```

현재는 행동 분류 성능까지 바로 보지 않아도 된다.

우선 확인할 것:

```text
1. MediaPipe skeleton 추출이 되는가?
2. skeleton.shape = (T, 33, 3)이 저장되는가?
3. VA-AFS가 raw beta와 online score를 계산하는가?
4. warmup, tau, k_max 이유로 frame이 선택되는가?
5. tau와 k_max에 따라 processed frame ratio가 변하는가?
6. selected_frames 폴더의 이미지가 plot의 selected_indices와 일치하는가?
```

---

## 15. BlockGCN으로 accuracy 확인

BlockGCN은 NTU `.npz`를 다음 shape로 읽는다.

```python
x_train.shape = (N, T, 150)
x_test.shape = (N, T, 150)
```

여기서 `150 = M(2) * V(25) * C(3)`이다. VA-AFS를 BlockGCN 앞단에 붙일 때는 선택된 frame만 앞쪽에 채우고 나머지는 zero padding으로 둔다. 이렇게 해야 BlockGCN feeder가 기존 방식대로 valid frame 수를 계산하고, 최종 입력 길이 64로 resize할 수 있다.

BlockGCN 실행에는 별도 의존성이 필요하다.

```bash
cd ../BlockGCN
pip install -e torchlight
pip install torch tensorboardX scikit-learn pyyaml
```

### 15.1 BlockGCN 원본 데이터 생성

데이터셋 다운로드가 끝나면 BlockGCN README 순서대로 NTU `.npz`를 만든다.

```bash
cd ../BlockGCN/data/ntu
python get_raw_skes_data.py
python get_raw_denoised_data.py
python seq_transformation.py
```

생성 예시:

```text
../BlockGCN/data/ntu/NTU60_CS.npz
../BlockGCN/data/ntu/NTU60_CV.npz
```

### 15.2 VA-AFS 적용 `.npz` 생성

`VA-AFS/` 폴더에서 실행한다.

```bash
python apply_va_afs_to_blockgcn_npz.py \
  --input_npz ../BlockGCN/data/ntu/NTU60_CS.npz \
  --tau 0.9 \
  --k_max 13 \
  --window_size 13
```

출력 예시:

```text
outputs/blockgcn_npz/NTU60_CS_vaafs_tau0.9_k13_w13.npz
```

이 파일에는 기존 `x_train`, `y_train`, `x_test`, `y_test`와 함께 다음 metadata가 추가된다.

```text
train_original_counts
train_selected_counts
train_processed_frame_ratio
test_original_counts
test_selected_counts
test_processed_frame_ratio
vaafs_tau
vaafs_k_max
vaafs_window_size
```

test split만 줄이고 train split은 그대로 두고 싶으면 다음처럼 실행한다.

```bash
python apply_va_afs_to_blockgcn_npz.py \
  --input_npz ../BlockGCN/data/ntu/NTU60_CS.npz \
  --output_npz outputs/blockgcn_npz/NTU60_CS_vaafs_test_only.npz \
  --splits test
```

### 15.3 BlockGCN accuracy 실행

BlockGCN pretrained weight 또는 직접 학습한 checkpoint가 필요하다. 준비되면 `VA-AFS/` 폴더에서 다음처럼 실행한다.

```bash
python run_blockgcn_acc.py \
  --data_npz outputs/blockgcn_npz/NTU60_CS_vaafs_tau0.9_k13_w13.npz \
  --weights ../BlockGCN/work_dir/ntu60/csub/your_checkpoint.pt \
  --config config/nturgbd-cross-subject/default.yaml \
  --device 0 \
  --test_batch_size 64 \
  --num_worker 4 \
  --save_score
```

wrapper가 하는 일:

```text
1. BlockGCN config를 읽는다.
2. test_feeder_args.data_path를 VA-AFS 적용 .npz로 바꾼 임시 config를 만든다.
3. ../BlockGCN/main.py --phase test를 실행한다.
4. stdout의 Accuracy 값을 읽어 Top-1 accuracy를 다시 출력한다.
```

실행 전 명령만 확인하려면:

```bash
python run_blockgcn_acc.py \
  --data_npz outputs/blockgcn_npz/NTU60_CS_vaafs_tau0.9_k13_w13.npz \
  --weights ../BlockGCN/work_dir/ntu60/csub/your_checkpoint.pt \
  --dry_run
```

비교 실험은 같은 checkpoint로 원본 `.npz`와 VA-AFS `.npz`를 각각 평가한다.

```text
Baseline acc: BlockGCN + original NTU60_CS.npz
VA-AFS acc  : BlockGCN + NTU60_CS_vaafs_tau*_k*_w*.npz
```

Device는 기본적으로 자동 선택된다.

```text
CUDA 가능 → --device 0
Apple Silicon MPS 가능 → --device mps
그 외 → --device cpu
```

따라서 Colab GPU 런타임에서는 `--device`를 생략해도 CUDA device 0으로 실행된다. 직접 지정하고 싶으면 다음처럼 넘긴다.

```bash
python run_blockgcn_acc.py \
  --data_npz outputs/blockgcn_npz/NTU60_CS_vaafs_tau0.9_k13_w13.npz \
  --weights ../BlockGCN/work_dir/ntu60/csub/your_checkpoint.pt \
  --device 0
```

Mac에서 MPS 문제가 생기면 CPU로 강제한다.

```bash
python run_blockgcn_acc.py \
  --data_npz outputs/blockgcn_npz/NTU60_CS_vaafs_tau0.9_k13_w13.npz \
  --weights ../BlockGCN/work_dir/ntu60/csub/your_checkpoint.pt \
  --device cpu
```

주의:

```text
CPU/MPS는 CUDA보다 느리다. 전체 NTU accuracy 평가는 오래 걸릴 수 있다.
MPS에서 지원되지 않는 PyTorch 연산이 있으면 --device cpu로 다시 실행한다.
최종 대규모 실험은 Colab, 연구실 GPU 서버, CUDA 가능한 데스크톱에서 돌리는 것이 현실적이다.
```

---

## 16. 자주 나는 오류

### `ModuleNotFoundError: No module named 'cv2'`

원인:

```text
opencv-python이 현재 가상환경에 설치되지 않음
```

해결:

```bash
python -m pip install opencv-python
```

확인:

```bash
python -c "import cv2; print(cv2.__version__)"
```

### `ModuleNotFoundError: No module named 'mediapipe'`

해결:

```bash
python -m pip install mediapipe
```

확인:

```bash
python -c "import mediapipe as mp; print(mp.__version__)"
```

### 영상 파일을 못 찾는 경우

실행 위치와 영상 경로를 확인한다.

```bash
pwd
ls videos
```

상대경로가 헷갈리면 절대경로를 사용한다.

```bash
python extract_skeleton_mediapipe.py \
  --video "/Users/minsukim/.../VA-AFS/videos/squat_01.mp4"
```

---

## 17. 발표에 넣을 설명

```text
Since public skeleton benchmark datasets may require access approval, the initial prototype is validated using self-recorded videos and MediaPipe-extracted skeleton sequences.
This allows us to verify whether VA-AFS selects motion-informative frames before applying it to large-scale benchmarks such as NTU RGB+D.
```

한국어:

```text
공개 skeleton benchmark dataset은 접근 승인이 필요할 수 있으므로, 초기 프로토타입은 직접 촬영한 영상과 MediaPipe로 추출한 skeleton sequence를 이용해 검증한다.
이를 통해 NTU RGB+D 같은 대규모 benchmark에 적용하기 전에 VA-AFS가 motion-informative frame을 선택하는지 먼저 확인한다.
```

---

## 18. 오늘 할 일 체크리스트

```text
1. requirements 설치
2. videos/ 폴더에 테스트 영상 넣기
3. extract_skeleton_mediapipe.py 실행
4. outputs/에 skeleton.npy 생성 확인
5. va_afs_threshold.py 실행
6. selection plot 확인
7. tau별 processed frame ratio 비교
```

완료 기준:

```text
outputs/
├── *_skeleton.npy
├── *_skeleton.csv
├── *_preview.mp4
├── *_sampled_*.npy
└── *_selection_*.png
```
