# Colab 실행 가이드

이 문서는 처음 보는 사람이 GitHub에서 이 프로젝트를 clone한 뒤, Google Colab 또는 VS Code Colab extension으로 발표용 실험을 실행하는 절차이다.

## 1. VS Code에서 Colab 연결

VS Code에서 `.ipynb` 파일을 하나 열고 우상단 `Select Kernel`을 누른다.

```text
Select Kernel -> Colab -> Auto Connect
```

이 저장소에는 같은 내용을 담은 `colab_run.ipynb`도 포함되어 있다. VS Code에서는 이 파일을 열고 Colab kernel을 선택한 뒤 셀을 순서대로 실행하면 된다.

Google 로그인 후 연결되면 아래 셀들을 순서대로 실행한다. Colab kernel은 로컬 파일을 직접 보지 못하므로, Colab 런타임 안의 `/content/src`에 GitHub repo를 다시 clone한다.

## 2. Drive mount와 repo clone

데이터는 아래 Google Drive 공유 폴더에서 받는다.

```text
https://drive.google.com/drive/folders/1JX_Jf2jayPkdh8ii4jwNCrZSe6KdweKh?usp=sharing
```

처음 실행하는 사람은 위 공유 폴더의 zip 파일 4개를 자기 Google Drive에 복사하거나 다운로드 후 업로드해서 아래 경로로 맞춘다.

```text
MyDrive/AFS/data/
├── all_sqe.zip
├── nturgbd_skeletons_s001_to_s017.zip
├── nturgbd_skeletons_s018_to_s032.zip
└── videos.zip
```

주의: Colab에서 `drive.mount("/content/drive")`를 하면 현재 로그인한 Google 계정의 Drive만 보인다. 내 Drive에 있는 파일은 다른 사람의 Colab 런타임에 자동으로 보이지 않으므로, 공유 링크로 접근 권한을 받은 뒤 자기 Drive에 복사하거나 shortcut/path를 맞춰야 한다.

노트북 첫 셀:

```python
from google.colab import drive
drive.mount("/content/drive")
```

두 번째 셀:

```bash
%cd /content
!git clone https://github.com/Kiim-Miin-Su/VA-AFS.git /content/src
%cd /content/src
```

이미 `/content/src`가 있으면 런타임을 다시 시작하거나 아래처럼 지우고 clone한다.

```bash
%cd /content
!rm -rf /content/src
!git clone https://github.com/Kiim-Miin-Su/VA-AFS.git /content/src
%cd /content/src
```

## 3. 데이터 압축 해제와 의존성 설치

Colab 메뉴에서 GPU 런타임이 켜져 있는지 확인한 뒤 실행한다.

```bash
!python setup_colab.py --data_dir /content/drive/MyDrive/AFS/data --install --verify
```

`setup_colab.py`는 zip을 올바른 위치에 풀고, NTU 폴더가 중첩된 경우 자동으로 한 단계 펴준다.

Drive 경로를 다르게 올렸다면 `--data_dir`만 바꾸면 된다. 예를 들어 `MyDrive/va-afs-data/`에 올렸다면:

```bash
!python setup_colab.py --data_dir /content/drive/MyDrive/va-afs-data --install --verify
```

## 4. 발표용 전체 파이프라인

subset 3000개, epoch 80 기준 실행:

```bash
!python VA-AFS/run_colab_pipeline.py \
  --sample_size 3000 \
  --num_epoch 80 \
  --batch_size 64 \
  --test_batch_size 64 \
  --num_worker 2
```

이 명령은 아래 작업을 순서대로 실행한다.

```text
1. NTU subset 폴더 생성
2. BlockGCN 전처리로 NTU60_CS.npz 생성
3. 원본 subset으로 BlockGCN 학습
4. test split에 VA-AFS 적용
5. 원본 subset과 VA-AFS subset accuracy 비교
```

## 5. 짧은 smoke test

런타임 확인만 빠르게 하려면 발표용 명령 대신 아래를 먼저 실행한다.

```bash
!python VA-AFS/run_colab_pipeline.py \
  --sample_size 200 \
  --num_epoch 2 \
  --batch_size 8 \
  --test_batch_size 8 \
  --num_worker 1
```

## 6. 다시 실행할 때

```bash
!python VA-AFS/run_colab_pipeline.py --force_subset --force_preprocess
!python VA-AFS/run_colab_pipeline.py --force_train
!python VA-AFS/run_colab_pipeline.py --force_vaafs
```

명령만 확인하려면:

```bash
!python VA-AFS/run_colab_pipeline.py --force_subset --force_preprocess --force_train --force_vaafs --dry_run
```

## 7. 결과 위치

```text
BlockGCN/data/ntu_subset_3000/NTU60_CS.npz
VA-AFS/outputs/blockgcn_train/ntu_subset_3000_original_e80/runs-*.pt
VA-AFS/outputs/blockgcn_npz/NTU60_CS_subset3000_vaafs_test_only_tau0.9_k13_w13.npz
VA-AFS/outputs/blockgcn_npz/NTU60_CS_subset3000_vaafs_test_only_tau0.9_k13_w13.summary.txt
VA-AFS/outputs/blockgcn_acc/ntu_subset_3000_original_e80/log.txt
VA-AFS/outputs/blockgcn_acc/ntu_subset_3000_vaafs_e80/log.txt
```

발표에서는 두 accuracy 로그와 `*.summary.txt`의 `processed frame ratio`, `frame reduction`을 같이 보여주면 된다.
