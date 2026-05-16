# Colab 실행 가이드

이 문서는 Google Colab 웹에서 `colab_run.ipynb`를 열어 VA-AFS 발표용 실험을 실행하는 절차이다. VS Code Colab extension에서도 같은 노트북을 순서대로 실행하면 된다.

## 1. Colab 웹에서 열기

GitHub에 올라간 노트북을 Colab에서 바로 연다.

```text
https://colab.research.google.com/github/Kiim-Miin-Su/VA-AFS/blob/main/colab_run.ipynb
```

Colab 메뉴에서 GPU 런타임을 먼저 선택한다.

```text
Runtime -> Change runtime type -> Hardware accelerator -> GPU
```

노트북 기본값은 긴 학습을 바로 시작하지 않도록 smoke test만 실행한다. 발표용 80 epoch 실험은 설정 셀에서 `RUN_PRESENTATION = True`로 바꾼 뒤 실행한다.

## 2. 데이터 준비

데이터는 아래 Google Drive 공유 폴더에서 받는다.

```text
https://drive.google.com/drive/folders/1JX_Jf2jayPkdh8ii4jwNCrZSe6KdweKh?usp=sharing
```

처음 실행하는 사람은 공유 폴더의 zip 파일 4개를 자기 Google Drive에 복사하거나 다운로드 후 업로드해서 아래 경로로 맞춘다.

```text
MyDrive/AFS/data/
├── all_sqe.zip
├── nturgbd_skeletons_s001_to_s017.zip
├── nturgbd_skeletons_s018_to_s032.zip
└── videos.zip
```

주의: Colab에서 `drive.mount("/content/drive")`를 하면 현재 로그인한 Google 계정의 Drive만 보인다. 공유 폴더에 접근 권한이 있어도 zip 파일이 내 Drive 경로에 없으면 `/content/drive/MyDrive/AFS/data`에서 찾을 수 없다.

다른 경로를 쓰는 경우 노트북 설정 셀의 `DATA_DIR`만 바꾼다.

```python
DATA_DIR = "/content/drive/MyDrive/va-afs-data"
```

## 3. 노트북 실행 흐름

`colab_run.ipynb`는 다음 순서로 구성되어 있다.

```text
1. Google Drive mount
2. 실행 설정
3. /content/src에 GitHub repo clone
4. Drive zip 파일 존재 확인
5. 데이터 압축 해제와 Colab 의존성 설치
6. GPU 확인
7. smoke test
8. 발표용 pipeline
9. 전체 NTU60 optional run
10. 결과 시각화
11. Drive backup optional
```

repo clone은 `/content/src`가 이미 있으면 재사용한다. 최신 GitHub 내용을 다시 받고 싶으면 설정 셀에서 `FORCE_RECLONE = True`로 바꾼다.

`setup_colab.py`는 이미 압축 해제된 데이터가 있으면 건너뛰고, Colab requirements import가 이미 가능한 경우 pip 설치도 건너뛴다.

## 4. 빠른 smoke test

기본 설정은 아래와 같다.

```python
RUN_SMOKE_TEST = True
SMOKE_SAMPLE_SIZE = 200
SMOKE_NUM_EPOCH = 2
SMOKE_BATCH_SIZE = 8
```

실행되는 명령은 다음과 같다.

```bash
python VA-AFS/run_colab_pipeline.py \
  --sample_size 200 \
  --num_epoch 2 \
  --batch_size 8 \
  --test_batch_size 8 \
  --num_worker 1
```

이 단계는 런타임, 데이터, dependency, BlockGCN 호출이 정상인지 확인하기 위한 것이다. accuracy는 의미 있게 해석하지 않는다.

## 5. 발표용 전체 파이프라인

발표용 실행은 설정 셀에서 아래처럼 바꾼다.

```python
RUN_SMOKE_TEST = False
RUN_PRESENTATION = True
SHOW_FIGURES = True

PRESENTATION_SAMPLE_SIZE = 3000
PRESENTATION_NUM_EPOCH = 80
PRESENTATION_BATCH_SIZE = 64
PRESENTATION_TEST_BATCH_SIZE = 64
PRESENTATION_NUM_WORKER = 2
```

실행되는 명령은 다음과 같다.

```bash
python VA-AFS/run_colab_pipeline.py \
  --sample_size 3000 \
  --num_epoch 80 \
  --batch_size 64 \
  --test_batch_size 64 \
  --num_worker 2
```

파이프라인은 아래 작업을 순서대로 실행한다.

```text
1. NTU subset 폴더 생성
2. BlockGCN 전처리로 NTU60_CS.npz 생성
3. 원본 subset으로 BlockGCN 학습
4. test split에 VA-AFS 적용
5. 원본 subset과 VA-AFS subset accuracy 비교
6. frame reduction/accuracy plot 생성
```

더 큰 실험이 필요하면 `PRESENTATION_SAMPLE_SIZE`를 `18000`처럼 늘린다. Colab 무료 GPU에서는 런타임 제한 때문에 먼저 3000개 subset으로 전체 흐름을 확인하는 편이 안전하다.

## 6. 다시 실행할 때

`run_colab_pipeline.py`는 이미 만들어진 subset, `.npz`, checkpoint, VA-AFS 결과를 기본적으로 재사용한다.

필요한 단계만 강제로 다시 만들 수 있다.

```bash
python VA-AFS/run_colab_pipeline.py --force_subset --force_preprocess
python VA-AFS/run_colab_pipeline.py --force_train
python VA-AFS/run_colab_pipeline.py --force_vaafs
```

발표 figure 셀에서 아래 에러가 나면 accuracy 평가 로그가 아직 없다는 뜻이다.

```text
FileNotFoundError: Accuracy logs were not found.
```

frame ratio plot은 VA-AFS `.npz`만 있으면 그려지지만, accuracy plot은 아래 두 로그가 필요하다.

```text
VA-AFS/outputs/blockgcn_acc/ntu_subset_<sample_size>_original_e<num_epoch>/log.txt
VA-AFS/outputs/blockgcn_acc/ntu_subset_<sample_size>_vaafs_e<num_epoch>/log.txt
```

이 경우 Colab에서 발표용 pipeline 셀을 같은 `PRESENTATION_SAMPLE_SIZE`, `PRESENTATION_NUM_EPOCH` 값으로 다시 실행한다. 이미 만들어진 subset, preprocessing 결과, checkpoint, VA-AFS `.npz`는 재사용되고 누락된 eval 로그가 다시 생성된다.

명령만 확인하려면:

```bash
python VA-AFS/run_colab_pipeline.py \
  --force_subset \
  --force_preprocess \
  --force_train \
  --force_vaafs \
  --dry_run
```

## 7. 결과 위치

`sample_size=3000`, `epoch=80` 기준 주요 결과는 아래에 생긴다.

```text
BlockGCN/data/ntu_subset_3000/NTU60_CS.npz
VA-AFS/outputs/blockgcn_train/ntu_subset_3000_original_e80/runs-*.pt
VA-AFS/outputs/blockgcn_npz/NTU60_CS_ntu_subset_3000_vaafs_test_only_tau0.9_k13_w13.npz
VA-AFS/outputs/blockgcn_npz/NTU60_CS_ntu_subset_3000_vaafs_test_only_tau0.9_k13_w13.summary.txt
VA-AFS/outputs/blockgcn_acc/ntu_subset_3000_original_e80/log.txt
VA-AFS/outputs/blockgcn_acc/ntu_subset_3000_vaafs_e80/log.txt
VA-AFS/outputs/presentation_plots/subset3000_test_frame_ratio.png
VA-AFS/outputs/presentation_plots/subset3000_e80_accuracy.png
```

Colab 런타임은 종료되면 `/content/src`의 결과가 사라진다. 결과를 Drive에 남기려면 설정 셀에서 `BACKUP_RESULTS_TO_DRIVE = True`로 바꿔 마지막 backup 셀을 실행한다.
