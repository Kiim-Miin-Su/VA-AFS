# VA-AFS Demo README

> **목표**  
> 직접 촬영한 영상에서 MediaPipe로 skeleton coordinate sequence를 추출하고, VA-AFS threshold gate를 적용해 중요한 skeleton frame만 선택한다.
> subset3000, epoch80 기준 Top1-acc: <>, Top5-acc: <>

---

## 데이터 다운로드

Google Drive: https://drive.google.com/drive/folders/1JX_Jf2jayPkdh8ii4jwNCrZSe6KdweKh?usp=sharing

Colab 웹에서 바로 실행하려면 `colab_run.ipynb`를 연 뒤 `Runtime -> Change runtime type -> GPU`를 선택한다. 기본 설정은 빠른 smoke test만 실행하므로, 발표용 학습은 노트북 설정 셀에서 `RUN_PRESENTATION = True`와 `SHOW_FIGURES = True`로 바꾼다.

Colab 또는 VS Code Colab extension에서 발표용으로 돌릴 때는 위 Google Drive 공유 폴더의 zip 파일 4개를 자기 Drive의 `MyDrive/AFS/data/`에 둔다. 그 다음 노트북을 순서대로 실행하거나 아래 명령을 실행한다.

```bash
python setup_colab.py --data_dir /content/drive/MyDrive/AFS/data --install --verify
python VA-AFS/run_colab_pipeline.py --sample_size 3000 --num_epoch 80 --batch_size 64 --test_batch_size 64 --num_worker 2
```

자세한 Colab 절차와 결과 파일 위치는 `COLAB.md`를 참고한다.
VS Code Colab extension에서는 `colab_run.ipynb`를 열고 `Select Kernel -> Colab -> Auto Connect`를 선택한 뒤 같은 설정 셀을 조정해 실행하면 된다.

이 저장소는 GitHub에 바로 올릴 수 있도록 코드, 설정 파일, 빈 폴더 구조만 포함한다. 대용량 원본 데이터와 실행 결과는 위 Google Drive에서 받은 뒤 `src/` 기준으로 아래 위치에 배치한다.

| 데이터 묶음                          | 압축 해제 위치               | 압축 해제 후 기대 경로                             |
| ------------------------------------ | ---------------------------- | -------------------------------------------------- |
| `all_sqe.zip`                        | `BlockGCN/data/NW-UCLA/`     | `BlockGCN/data/NW-UCLA/all_sqe/`                   |
| `nturgbd_skeletons_s001_to_s017.zip` | `BlockGCN/data/nturgbd_raw/` | `BlockGCN/data/nturgbd_raw/nturgb+d_skeletons/`    |
| `nturgbd_skeletons_s018_to_s032.zip` | `BlockGCN/data/nturgbd_raw/` | `BlockGCN/data/nturgbd_raw/nturgb+d_skeletons120/` |
| `videos.zip`                         | `VA-AFS/`                    | `VA-AFS/videos/`                                   |

가장 쉬운 복원 방법은 다운로드한 zip 파일들을 `src/../data/`에 둔 뒤, `src/` 폴더에서 아래 명령을 그대로 실행하는 것이다.

```bash
unzip ../data/all_sqe.zip -d BlockGCN/data/NW-UCLA
unzip ../data/nturgbd_skeletons_s001_to_s017.zip -d BlockGCN/data/nturgbd_raw
unzip ../data/nturgbd_skeletons_s018_to_s032.zip -d BlockGCN/data/nturgbd_raw
unzip ../data/videos.zip -d VA-AFS
```

전체 NTU60/NTU120 전처리를 하려면 `nturgbd_skeletons_s001_to_s017.zip`과 `nturgbd_skeletons_s018_to_s032.zip`이 모두 필요하다. 데이터 복원 후에는 BlockGCN README 순서대로 `data/ntu` 또는 `data/ntu120`의 전처리 스크립트를 실행해 `.npz`를 생성한다.

주의: NTU zip 파일은 압축 내부에 `nturgb+d_skeletons/` 또는 `nturgb+d_skeletons120/` 폴더를 이미 포함한다. 따라서 `BlockGCN/data/nturgbd_raw/nturgb+d_skeletons` 안에 다시 풀면 다음처럼 중첩된다.

```text
BlockGCN/data/nturgbd_raw/nturgb+d_skeletons/nturgb+d_skeletons/*.skeleton
```

이 구조는 BlockGCN 전처리 스크립트가 기대하는 구조가 아니다. 잘못 풀었다면 안쪽 파일을 바깥 폴더로 옮겨서 아래처럼 만든다.

```text
BlockGCN/data/nturgbd_raw/nturgb+d_skeletons/*.skeleton
BlockGCN/data/nturgbd_raw/nturgb+d_skeletons120/*.skeleton
```

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

저장소에 기본으로 유지하는 구조는 코드, 모델 파일, 빈 출력 폴더, 입력 영상 폴더이다. 대용량 데이터와 실행 결과는 Git에 올리지 않는다.

```text
VA-AFS/
├── videos/
│   ├── .gitkeep
│   └── *.avi / *.mp4          # 외부 데이터 복원 후 생기는 입력 영상
├── models/
│   └── pose_landmarker_lite.task
├── outputs/
│   ├── .gitkeep
│   ├── skeleton/
│   ├── previews/
│   ├── threshold/
│   ├── blockgcn_npz/
│   ├── blockgcn_configs/
│   ├── blockgcn_train/
│   └── blockgcn_acc/
├── extract_skeleton_mediapipe.py
├── va_afs_threshold.py
├── prepare_ntu_subset.py
├── apply_va_afs_to_blockgcn_npz.py
├── run_blockgcn_train.py
├── run_blockgcn_acc.py
├── keep_best_blockgcn_checkpoint.py
├── constants.py
└── README.md
```

실행 후에는 입력 파일 이름에 맞춰 아래와 같은 결과가 생성된다. 예시는 `videos/ido_run.avi`를 처리한 경우이다.

```text
outputs/
├── skeleton/
│   ├── npy/
│   │   └── ido_run_skeleton.npy
│   └── csv/
│       └── ido_run_skeleton.csv
├── previews/
│   └── mp4/
│       └── ido_run_preview.mp4
└── threshold/
    ├── npy/
    │   └── ido_run_skeleton_tau0.9_k13_sampled.npy
    ├── plots/
    │   └── ido_run_skeleton_tau0.9_k13_selection.png
    └── selected_frames/
        └── ido_run_skeleton_tau0.9_k13/
            ├── ido_run_frame_000000.jpg
            └── ido_run_frame_000013.jpg
```

출력은 기능과 확장자 기준으로 나뉜다.

| 경로                                 | 의미                                   |
| ------------------------------------ | -------------------------------------- |
| `outputs/skeleton/npy/`              | MediaPipe skeleton `.npy`              |
| `outputs/skeleton/csv/`              | frame/joint 좌표 CSV                   |
| `outputs/previews/mp4/`              | pose landmark preview video            |
| `outputs/threshold/npy/`             | VA-AFS가 선택한 skeleton sequence      |
| `outputs/threshold/plots/`           | 선택 과정 시각화 plot                  |
| `outputs/threshold/selected_frames/` | 실제 선택된 원본 frame 이미지          |
| `outputs/blockgcn_npz/`              | BlockGCN 평가용 VA-AFS 적용 NTU `.npz` |
| `outputs/blockgcn_configs/`          | BlockGCN 실행용 임시 config            |
| `outputs/blockgcn_train/`            | BlockGCN subset 학습 로그와 checkpoint |
| `outputs/blockgcn_acc/`              | BlockGCN accuracy 로그와 score         |

### 2.1 대용량 데이터 복원

GitHub에는 실행에 필요한 폴더 구조와 코드만 유지하고, 대용량 데이터와 생성 결과 파일은 올리지 않는다. 저장소에는 빈 데이터 폴더를 보존하기 위한 `.gitkeep`, BlockGCN 전처리에 필요한 작은 `.py`, `statistics/*.txt`, 그리고 NW-UCLA ensemble에서 참조하는 `BlockGCN/data/NW-UCLA/val_label.pkl`만 포함된다.

상단의 Google Drive 데이터 묶음 또는 원본 데이터셋을 내려받은 뒤, `src/` 기준으로 아래 경로가 채워지게 한다.

```text
BlockGCN/data/NW-UCLA/all_sqe/
BlockGCN/data/nturgbd_raw/nturgb+d_skeletons/
BlockGCN/data/nturgbd_raw/nturgb+d_skeletons120/
VA-AFS/videos/
```

압축 파일을 `../data/`에 받은 경우 예시는 다음과 같다.

```bash
unzip ../data/all_sqe.zip -d BlockGCN/data/NW-UCLA
unzip ../data/nturgbd_skeletons_s001_to_s017.zip -d BlockGCN/data/nturgbd_raw
unzip ../data/nturgbd_skeletons_s018_to_s032.zip -d BlockGCN/data/nturgbd_raw
unzip ../data/videos.zip -d VA-AFS
```

복원 후 확인:

```bash
ls BlockGCN/data/NW-UCLA/all_sqe
ls BlockGCN/data/nturgbd_raw
ls VA-AFS/videos
```

주의:

```text
BlockGCN/data 아래의 대용량 raw/json/npz 데이터 파일, VA-AFS/videos, VA-AFS/outputs, outputs 아래의 실제 데이터 파일은 .gitignore 대상이다.
단, BlockGCN의 `ensemble.py`가 직접 읽는 `BlockGCN/data/NW-UCLA/val_label.pkl`은 작은 라벨 메타데이터이므로 Git에 유지한다.
새 데이터를 추가해도 GitHub에는 올라가지 않는다.
폴더 구조는 .gitkeep으로 유지한다.
```

### 2.2 BlockGCN 통합 상태

`BlockGCN/`은 공식 BlockGCN 구현을 이 프로젝트 안에 vendor 형태로 둔 것이다. 공식 저장소의 최상위 구성인 `config/`, `data/`, `feeders/`, `graph/`, `model/`, `torchlight/`, `main.py`, `train.sh`, `evaluate.sh`, `ensemble.py`는 유지한다.

다만 이 프로젝트의 `VA-AFS` 코드는 `../BlockGCN`을 직접 호출해서 accuracy 평가를 수행한다. 따라서 `BlockGCN/`을 원본 저장소로 다시 덮어쓸 때도 다음 로컬 정책은 유지해야 한다.

```text
1. BlockGCN 소스/설정/전처리 스크립트는 유지한다.
2. BlockGCN/data의 대용량 raw 데이터와 생성 npz는 Git에 올리지 않는다.
3. VA-AFS/videos와 VA-AFS/outputs의 입력/출력 파일도 Git에 올리지 않는다.
4. VA-AFS의 run_blockgcn_acc.py, apply_va_afs_to_blockgcn_npz.py가 참조하는 상대 경로를 바꾸지 않는다.
5. NW-UCLA ensemble에 필요한 BlockGCN/data/NW-UCLA/val_label.pkl은 유지한다.
```

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
PyYAML
```

설치:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt`가 아직 없다면 직접 설치한다.

```bash
python -m pip install opencv-python mediapipe numpy pandas matplotlib tqdm PyYAML
```

설치 확인:

```bash
python -c "import cv2, mediapipe, numpy, pandas, matplotlib, tqdm, yaml; print('ok')"
```

---

## 5. 영상 준비

`videos/` 폴더에 테스트 영상을 넣는다. 아래는 외부 데이터 복원 후 존재하는 `ido_run.avi`를 사용하는 예시이다. 다른 영상을 쓰면 파일명만 바꾼다.

```text
videos/ido_run.avi
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
python extract_skeleton_mediapipe.py --video videos/ido_run.avi
```

다른 위치에서 실행한다면 전체 경로를 사용한다.

```bash
python /path/to/VA-AFS/extract_skeleton_mediapipe.py \
  --video /path/to/VA-AFS/videos/ido_run.avi
```

출력 파일:

```text
outputs/skeleton/npy/ido_run_skeleton.npy
outputs/skeleton/csv/ido_run_skeleton.csv
outputs/previews/mp4/ido_run_preview.mp4
```

| 파일             | 의미                                         |
| ---------------- | -------------------------------------------- |
| `*_skeleton.npy` | VA-AFS에 사용할 skeleton numpy 파일          |
| `*_skeleton.csv` | frame, joint, x, y, visibility를 저장한 CSV  |
| `*_preview.mp4`  | MediaPipe pose landmark가 그려진 확인용 영상 |

---

## 7. 추출된 skeleton shape

MediaPipe Pose 기준:

```python
skeleton.shape = (T, 33, 3)
```

| 축   | 의미                         |
| ---- | ---------------------------- |
| `T`  | 영상 frame 수                |
| `33` | MediaPipe Pose landmark 개수 |
| `3`  | `x`, `y`, `visibility`       |

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
  --npy outputs/skeleton/npy/ido_run_skeleton.npy
```

옵션을 직접 지정하는 예시:

```bash
python va_afs_threshold.py \
  --npy outputs/skeleton/npy/ido_run_skeleton.npy \
  --video videos/ido_run.avi \
  --tau 0.9 \
  --k_max 13 \
  --window_size 13
```

| 인자            | 의미                                       |                 기본값 |
| --------------- | ------------------------------------------ | ---------------------: |
| `--npy`         | MediaPipe로 추출한 skeleton `.npy` 경로    |                   필수 |
| `--video`       | 선택 frame 이미지를 저장할 원본 영상 경로  |              자동 탐색 |
| `--tau`         | 0~1 online score threshold                 |        `TAU[-1] = 0.9` |
| `--k_max`       | 최대 연속 skip 허용 frame 수               |       `K_MAX[-1] = 13` |
| `--window_size` | motion feature 계산에 사용할 최근 frame 수 | `WINDOW_SIZE[-1] = 13` |

`--video`를 생략하면 `.npy` 이름에서 `_skeleton`을 제거한 뒤 `videos/` 아래에서 같은 이름의 영상을 찾는다. 예를 들어 `ido_run_skeleton.npy`는 `videos/ido_run.avi`, `videos/ido_run.mp4` 등을 탐색한다.

---

## 9. window_size와 warmup 의미

`window_size`는 두 군데에서 쓰인다.

| 용도                  | 설명                                                                                                                            |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Motion feature window | 현재 frame의 displacement, velocity, acceleration, window 내부 표준편차를 계산할 때 최근 최대 `window_size`개 frame을 참조한다. |
| Warmup length         | online beta 평균/표준편차가 안정되기 전까지 처음 `window_size`개 frame을 강제로 선택한다.                                       |

`window_size = 13`이면 매 frame마다 최근 최대 13개의 frame을 보고 motion feature를 계산한다.

```text
T = 60
window_size = 13
```

| 현재 frame t | 참조 window |
| -----------: | ----------- |
|            0 | 0           |
|            1 | 0~1         |
|            2 | 0~2         |
|           12 | 0~12        |
|           13 | 1~13        |
|           14 | 2~14        |
|          ... | ...         |
|           59 | 47~59       |

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
Saved sampled skeleton: outputs/threshold/npy/ido_run_skeleton_tau0.9_k13_sampled.npy
Saved plot: outputs/threshold/plots/ido_run_skeleton_tau0.9_k13_selection.png
Saved selected frame images: 15 files
Selected frame image dir: outputs/threshold/selected_frames/ido_run_skeleton_tau0.9_k13
```

| 출력                          | 의미                                           |
| ----------------------------- | ---------------------------------------------- |
| `Original shape`              | 원본 skeleton shape                            |
| `Sampled shape`               | 선택된 skeleton shape                          |
| `Processed frame ratio`       | 전체 frame 중 선택된 frame 비율                |
| `Selected indices`            | 선택된 frame 번호                              |
| `Saved selected frame images` | 실제 원본 영상에서 저장한 선택 frame 이미지 수 |

---

## 12. 결과 파일

예시:

```text
outputs/threshold/npy/ido_run_skeleton_tau0.9_k13_sampled.npy
outputs/threshold/plots/ido_run_skeleton_tau0.9_k13_selection.png
outputs/threshold/selected_frames/ido_run_skeleton_tau0.9_k13/ido_run_frame_000000.jpg
```

| 파일              | 의미                                                     |
| ----------------- | -------------------------------------------------------- |
| `*_sampled.npy`   | 선택된 skeleton sequence                                 |
| `*_selection.png` | online score, raw beta, motion change, 선택 frame 시각화 |
| `*_frame_*.jpg`   | 실제 선택된 원본 frame 이미지                            |

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
python va_afs_threshold.py --npy outputs/skeleton/npy/ido_run_skeleton.npy --tau 0.5 --k_max 13 --window_size 13
python va_afs_threshold.py --npy outputs/skeleton/npy/ido_run_skeleton.npy --tau 0.7 --k_max 13 --window_size 13
python va_afs_threshold.py --npy outputs/skeleton/npy/ido_run_skeleton.npy --tau 0.9 --k_max 13 --window_size 13
```

`k_max`도 바꿔본다.

```bash
python va_afs_threshold.py --npy outputs/skeleton/npy/ido_run_skeleton.npy --tau 0.9 --k_max 5 --window_size 13
python va_afs_threshold.py --npy outputs/skeleton/npy/ido_run_skeleton.npy --tau 0.9 --k_max 8 --window_size 13
python va_afs_threshold.py --npy outputs/skeleton/npy/ido_run_skeleton.npy --tau 0.9 --k_max 13 --window_size 13
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

| 기호   | 의미            |
| ------ | --------------- |
| \(T\)  | 원본 frame 수   |
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

BlockGCN accuracy 실험에서는 원본 영상이나 MediaPipe `.npy`를 직접 쓰지 않는다. 입력은 BlockGCN 전처리로 만든 NTU `.npz`이다.

두 실행 경로를 구분해야 한다.

| 목적                                  | 입력 파일                                   | 실행 스크립트                     | 결과                                              |
| ------------------------------------- | ------------------------------------------- | --------------------------------- | ------------------------------------------------- |
| 직접 촬영한 영상에서 VA-AFS 동작 확인 | `outputs/skeleton/npy/*_skeleton.npy`       | `va_afs_threshold.py`             | sampled `.npy`, 선택 frame 이미지, selection plot |
| BlockGCN accuracy 비교                | `BlockGCN/data/ntu_subset_200/NTU60_CS.npz` | `apply_va_afs_to_blockgcn_npz.py` | VA-AFS 적용 BlockGCN `.npz`                       |

`va_afs_threshold.py`는 `.npy`를 받는 단일 sequence 확인용 CLI이다. 반면 `apply_va_afs_to_blockgcn_npz.py`는 `.npz` 안의 `x_train`, `x_test` sequence를 하나씩 꺼내서 내부적으로 같은 `va_afs_threshold_sampling()` 함수를 호출한다.

BlockGCN은 NTU `.npz`를 다음 shape로 읽는다.

```python
x_train.shape = (N, T, 150)
x_test.shape = (N, T, 150)
```

여기서 `150 = M(2) * V(25) * C(3)`이다. VA-AFS를 BlockGCN 앞단에 붙일 때는 선택된 frame만 앞쪽에 채우고 나머지는 zero padding으로 둔다. 이렇게 해야 BlockGCN feeder가 기존 방식대로 valid frame 수를 계산하고, 최종 입력 길이 64로 resize할 수 있다.

따라서 `.skeleton` 원본 파일을 직접 줄이는 방식은 권장하지 않는다.

```text
권장하지 않음: raw .skeleton 파일에서 frame line을 직접 삭제
권장 방식    : raw .skeleton -> BlockGCN .npz 생성 -> .npz sequence에서 VA-AFS 적용
```

BlockGCN 실행에는 별도 의존성이 필요하다.

```bash
python -m pip install torch tensorboardX scikit-learn PyYAML einops torch-topological
```

이 프로젝트의 `run_blockgcn_train.py`와 `run_blockgcn_acc.py`는 subprocess 실행 시 `BlockGCN/torchlight`를 `PYTHONPATH`에 자동으로 넣는다. 따라서 보통 `pip install -e torchlight`를 따로 하지 않아도 된다.

### 15.1 200개 subset 필수 체크리스트

200개 subset만 사용할 경우 아래 단계만 실행하면 된다. 전체 NTU60 `.npz`를 만드는 15.6은 건너뛴다.

```text
[ ] 1. NTU raw skeleton zip을 BlockGCN/data/nturgbd_raw/ 아래에 압축 해제
[ ] 2. 200개 subset 전처리 폴더 생성
[ ] 3. subset 폴더에서 raw -> denoised -> npz 전처리 실행
[ ] 4. subset 원본 npz로 BlockGCN 학습
[ ] 5. subset test split에 VA-AFS 적용 npz 생성
[ ] 6. 같은 checkpoint로 원본 subset과 VA-AFS subset 평가
```

필수 산출물:

```text
BlockGCN/data/ntu_subset_200/NTU60_CS.npz
outputs/blockgcn_train/ntu_subset_200_original/runs-*.pt
outputs/blockgcn_npz/NTU60_CS_subset200_vaafs_test_only.npz
```

선택 산출물:

```text
BlockGCN/data/ntu/NTU60_CS.npz
```

위 선택 산출물은 전체 NTU60 benchmark를 돌릴 때만 필요하다.

### 15.2 200개 subset 생성

`VA-AFS/` 폴더에서 실행한다.

```bash
python prepare_ntu_subset.py \
  --sample_size 200 \
  --test_ratio 0.2 \
  --seed 1 \
  --sampling_strategy balanced_fallback_random \
  --output_dir ../BlockGCN/data/ntu_subset_200 \
  --force
```

이 스크립트는 원본 NTU60 `statistics/*.txt`에서 CS train/test performer pool을 기준으로 subset을 뽑고, 별도 전처리 폴더를 만든다. 기본 권장값인 `--sampling_strategy balanced_fallback_random`은 split별 클래스를 최대한 고르게 맞추되, exact balancing이 어려우면 같은 seed 기준의 완전 랜덤 샘플링으로 자동 fallback한다.

샘플링 전략:

- `balanced`: train/test split 안에서 클래스 수를 최대한 고르게 맞춘다.
- `random`: 클래스 라벨을 보지 않고 CS train/test pool에서 무작위 추출한다.
- `balanced_fallback_random`: exact per-class balancing이 가능하면 balanced를 쓰고, 불가능하면 random으로 fallback한다.

예를 들어 sample 수가 클래스 수와 잘 나누어떨어지고 각 클래스에 충분한 표본이 있으면 balanced가 사용된다. 반대로 exact balancing이 어려운 설정에서는 fallback 정책이 random을 사용한다.

```text
BlockGCN/data/ntu_subset_200/
├── statistics/
├── get_raw_skes_data.py
├── get_raw_denoised_data.py
└── seq_transformation.py
```

그 다음 subset 폴더에서 BlockGCN 전처리를 실행한다.

```bash
cd ../BlockGCN/data/ntu_subset_200
python get_raw_skes_data.py
python get_raw_denoised_data.py
python seq_transformation.py
```

생성 결과:

```text
../BlockGCN/data/ntu_subset_200/NTU60_CS.npz
../BlockGCN/data/ntu_subset_200/NTU60_CV.npz
```

작은 subset은 빠른 동작 확인용이다. 샘플 수가 작을수록 accuracy는 통계적으로 불안정하고, 클래스 분포도 전체 NTU60을 대표하지 않는다.

클래스 분포를 가장 엄격하게 통제하고 싶으면 `--samples_per_class_per_split`를 쓰는 편이 더 직접적이다. 예를 들어 `--samples_per_class_per_split 10`이면 CS train과 CS test에서 각 클래스당 최대 10개씩 고정 선택한다.

### 15.3 Subset 원본 학습

subset `.npz`가 생기면 작은 epoch로 먼저 학습이 되는지 확인한다.

```bash
cd ../../../VA-AFS

(mac)
python run_blockgcn_train.py \
  --data_npz ../BlockGCN/data/ntu_subset_200/NTU60_CS.npz \
  --work_dir outputs/blockgcn_train/ntu_subset_200_original \
  --num_epoch 30 \
  --batch_size 16 \
  --test_batch_size 16 \
  --num_worker 2 \
  --save_epoch 0 \
  --keep_best_only \
  --device mps
```

주의:

```text
BlockGCN train loader는 drop_last=True를 사용한다.
따라서 train sample 수가 batch_size보다 작으면 학습 batch가 0개가 된다.
작은 subset에서는 --batch_size를 4 또는 8처럼 작게 둔다.
BlockGCN 원본 save_epoch 기본값은 10이라 1~5 epoch 짧은 실험에서는 checkpoint가 저장되지 않는다.
이 wrapper는 기본값으로 --save_epoch 0을 사용해서 짧은 subset 학습에서도 checkpoint를 저장한다.
`--keep_best_only`를 붙이면 학습 완료 후 best epoch checkpoint만 남기고 나머지 `.pt` 파일은 삭제한다.
```

학습이 끝나면 `outputs/blockgcn_train/ntu_subset_200_original/` 아래에 checkpoint가 저장된다.

```text
outputs/blockgcn_train/ntu_subset_200_original/runs-*.pt
```

이 checkpoint를 원본 subset과 VA-AFS subset 평가에 동일하게 사용한다.
이미 여러 checkpoint가 쌓여 있으면 다음 명령으로 best checkpoint만 남길 수 있다.

```bash
python keep_best_blockgcn_checkpoint.py \
  outputs/blockgcn_train/ntu_subset_200_original
```

### 15.4 200개 subset에 VA-AFS 적용

`VA-AFS/` 폴더에서 실행한다. 기본 실험은 train split은 그대로 두고 test split만 VA-AFS로 줄이는 것이다.

```bash
python apply_va_afs_to_blockgcn_npz.py \
  --input_npz ../BlockGCN/data/ntu_subset_200/NTU60_CS.npz \
  --output_npz outputs/blockgcn_npz/NTU60_CS_subset200_vaafs_test_only.npz \
  --tau 0.9 \
  --k_max 13 \
  --window_size 13 \
  --splits test
```

출력 파일에는 기존 `x_train`, `y_train`, `x_test`, `y_test`와 함께 다음 metadata가 추가된다.

```text
test_original_counts
test_selected_counts
test_processed_frame_ratio
test_frame_reduction_ratio
vaafs_tau
vaafs_k_max
vaafs_window_size
```

`test_processed_frame_ratio`는 test sequence에서 실제로 남긴 frame 비율이다.
실행하면 콘솔과 `*.summary.txt`에 다음과 같은 요약도 저장된다.

```text
[test] VA-AFS frame selection summary
  samples                 : 40
  non-empty samples       : 40
  original frames total   : 3636
  selected frames total   : 1046
  processed frame ratio   : 0.288 (28.8%)
  frame reduction         : 0.712 (71.2%)
  original frame count    : mean=90.900, min=52.000, p25=66.750, median=78.500, p75=103.000, max=239.000
  selected frame count    : mean=26.150, min=15.000, p25=19.750, median=24.000, p75=30.000, max=70.000
  per-sample ratio        : mean=0.291, min=0.224, p25=0.275, median=0.290, p75=0.307, max=0.353
  ratio histogram         :
    0.0-0.2:    0
    0.2-0.4:   40 ########################
    0.4-0.6:    0
    0.6-0.8:    0
    0.8-1.0:    0
  first 8 samples      : 30/119, 24/75, 30/101, 22/80, 30/103, 15/52, 20/67, 20/60
```

요약 파일 예시:

```text
outputs/blockgcn_npz/NTU60_CS_subset200_vaafs_test_only.summary.txt
```

### 15.5 200개 subset accuracy 비교

먼저 checkpoint 파일명을 확인한다.

```bash
ls outputs/blockgcn_train/ntu_subset_200_original/*.pt
```

원본 subset 평가:

```bash
python run_blockgcn_acc.py \
  --data_npz ../BlockGCN/data/ntu_subset_200/NTU60_CS.npz \
  --weights outputs/blockgcn_train/ntu_subset_200_original/runs-*.pt \
  --test_batch_size 8 \
  --num_worker 1 \
  --save_score
```

VA-AFS test-only subset 평가:

```bash
python run_blockgcn_acc.py \
  --data_npz outputs/blockgcn_npz/NTU60_CS_subset200_vaafs_test_only.npz \
  --weights outputs/blockgcn_train/ntu_subset_200_original/runs-*.pt \
  --test_batch_size 8 \
  --num_worker 1 \
  --save_score
```

봐야 할 값:

```text
원본 subset acc
VA-AFS sampled subset acc
test_processed_frame_ratio
```

실행 전 명령만 확인하려면 `--dry_run`을 붙인다.

```bash
python run_blockgcn_acc.py \
  --data_npz outputs/blockgcn_npz/NTU60_CS_subset200_vaafs_test_only.npz \
  --weights outputs/blockgcn_train/ntu_subset_200_original/runs-*.pt \
  --dry_run
```

Device는 기본적으로 자동 선택된다.

```text
CUDA 가능 → --device 0
Apple Silicon MPS 가능 → --device mps
그 외 → --device cpu
```

Mac에서 MPS 문제가 생기면 CPU로 강제한다.

```bash
python run_blockgcn_train.py \
  --data_npz ../BlockGCN/data/ntu_subset_200/NTU60_CS.npz \
  --work_dir outputs/blockgcn_train/ntu_subset_200_original \
  --num_epoch 5 \
  --batch_size 8 \
  --test_batch_size 8 \
  --num_worker 1 \
  --device cpu \
  --save_epoch 0 \
  --keep_best_only
```

### 15.6 선택: 전체 NTU60 benchmark

전체 benchmark를 돌릴 수 있는 환경이면 BlockGCN README 순서대로 전체 NTU `.npz`를 만든다. 200개 subset 실험만 할 때는 이 단계가 필요 없다.

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

전체 NTU60에 VA-AFS를 적용하는 예시:

```bash
python apply_va_afs_to_blockgcn_npz.py \
  --input_npz ../BlockGCN/data/ntu/NTU60_CS.npz \
  --output_npz outputs/blockgcn_npz/NTU60_CS_vaafs_test_only.npz \
  --tau 0.9 \
  --k_max 13 \
  --window_size 13 \
  --splits test
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

### `ModuleNotFoundError: No module named 'yaml'`

원인:

```text
PyYAML이 설치되지 않았거나, 현재 터미널의 python이 가상환경 python이 아님
```

해결:

```bash
python -m pip install PyYAML
python -c "import yaml; print(yaml.__version__)"
```

`.venv`가 켜져 있는데도 계속 실패하면 `python` alias를 확인한다.

```bash
which python
python -c "import sys; print(sys.executable)"
```

정상 경로는 `src/.venv/bin/python`이어야 한다. 만약 Homebrew Python 같은 다른 경로가 나오면 현재 셸에서 alias를 해제한다.

```bash
unalias python
hash -r
```

### `ModuleNotFoundError: No module named 'einops'`

원인:

```text
BlockGCN/model/BlockGCN.py가 einops를 사용하지만 현재 가상환경에 설치되어 있지 않음
```

해결:

```bash
python -m pip install einops
```

### `ModuleNotFoundError: No module named 'torch_topological'`

원인:

```text
BlockGCN/model/BlockGCN.py가 torch_topological을 사용하지만 현재 가상환경에 설치되어 있지 않음
```

해결:

```bash
python -m pip install torch-topological
```

### `ImportError: cannot import name 'DictAction' from 'torchlight'`

원인:

```text
BlockGCN/torchlight 폴더 구조 때문에 Python이 실제 torchlight package 대신 바깥 namespace 폴더를 먼저 잡음
```

해결:

```text
run_blockgcn_train.py와 run_blockgcn_acc.py는 subprocess 실행 시 BlockGCN/torchlight를 PYTHONPATH에 자동 추가한다.
따라서 이 wrapper를 통해 실행하면 된다.
BlockGCN/main.py를 직접 실행할 때만 PYTHONPATH를 직접 지정한다.
```

직접 실행 예시:

```bash
PYTHONPATH=./torchlight python main.py --phase train --config path/to/config.yaml
```

### MPS에서 학습이 멈추거나 연산 에러가 나는 경우

Apple Silicon에서는 자동으로 `--device mps`가 선택될 수 있다. MPS에서 지원되지 않는 연산이 나오면 CPU로 강제한다.

```bash
python run_blockgcn_train.py \
  --data_npz ../BlockGCN/data/ntu_subset_200/NTU60_CS.npz \
  --work_dir outputs/blockgcn_train/ntu_subset_200_original \
  --num_epoch 5 \
  --batch_size 8 \
  --test_batch_size 8 \
  --num_worker 1 \
  --device cpu
```

### `TypeError: only 0-dimensional arrays can be converted to Python scalars`

원인:

```text
BlockGCN/feeders/tools.py의 valid_crop_resize()에서 p_interval=[0.5, 1] 구간 샘플링 결과를 길이 1짜리 NumPy array로 만든 뒤 int(...) 변환을 시도함
```

해결:

```text
p = float(np.random.uniform(p_interval[0], p_interval[1]))
```

이 프로젝트의 `BlockGCN/feeders/tools.py`는 위 방식으로 수정되어 있다. 새 BlockGCN 원본 코드로 덮어썼다면 같은 패치를 다시 적용해야 한다.

### 영상 파일을 못 찾는 경우

실행 위치와 영상 경로를 확인한다.

```bash
pwd
ls videos
```

상대경로가 헷갈리면 절대경로를 사용한다.

```bash
python extract_skeleton_mediapipe.py \
  --video "/Users/minsukim/.../VA-AFS/videos/ido_run.avi"
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
