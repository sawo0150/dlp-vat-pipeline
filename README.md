# DLP VAT Pipeline Dataset Generation Guide

이 문서는 `dlp-vat-pipeline` 레포지토리를 이용해  
**binary mask dataset 생성 → gray mask 생성 → projector window 생성 → LD 취득 → pairing → preprocess → train pack 추출**까지의 전체 과정을 정리한 가이드입니다.

---

## 1. Environment Setup

### 1-1. Miniconda 설치
먼저 `miniconda`를 설치합니다.

### 1-2. Repository clone

```bash
cd path/to/your/workspace
git clone https://github.com/sawo0150/dlp-vat-pipeline.git
cd dlp-vat-pipeline
````

### 1-3. Conda environment 생성

```bash
conda env create -f environment.yml
conda activate dlp
```

### 1-4. ImageNet 다운로드

```bash
python down_imagenet.py
```

#### 주의사항

`down_imagenet.py`를 실행하려면 Hugging Face token이 필요합니다.
스크립트 내부에서 아래 부분을 수정해야 합니다.

```python
from huggingface_hub import login  # 이 줄 추가

# 여기에 발급받은 token 입력
# login(token="hf_xxxxxx")

# 저장 경로 설정
OUTPUT_DIR = "../processed_imagenet"
```

---

## 2. Overall Pipeline

전체 데이터셋 생성 파이프라인은 아래 순서로 진행됩니다.

1. **Binary mask dataset 생성**
2. **Gray mask / boundary band 생성**
3. **Projector용 1080p window 생성**
4. **LabVIEW + CMOS를 이용한 light distribution(LD) 취득**
5. **Mask / LD pairing**
6. **Registration + preprocess**
7. **Train pack 추출 (optional)**

---

## 3. Step-by-Step Guide

---

### Step A. Binary Mask Dataset Generation

`run_binarydataset_gen.sh`는 `task=generate`를 실행하여
B1 ~ B5의 다섯 종류 binary dataset을 생성합니다.

실행:

```bash
./scripts/run_binarydataset_gen.sh
```

### 생성되는 dataset 종류

* **B1_Shape_Cutout**
  기본 도형(shape) 위주 + cutout + boundary speckle

* **B2_Grid_hf**
  spacing 2~7, thickness 2~4 중심의 고주파(high-frequency) grid 패턴

* **B3_Stripe_phase**
  period / duty / waviness를 가지는 stripe 패턴

* **B4_mix_Comp**
  shape / grid / stripe / imagenet 레이어를 2~3개 섞은 복합 패턴

* **B5_imagenet**
  ImageNet 기반 이진화 binary mask 패턴

### 참고사항

* 데이터 생성 비율은 대략 **B1 : B2 : B3 : B4 : B5 = 10 : 5 : 5 : 5 : 5**
* 스크립트 주석에는 기본값이 `1000`으로 적혀있어서 총`30000`개의 이미지가 생성됩니다.

---

### Step B. Gray Mask / Boundary Band Generation

`run_graymask_for_binarysets.sh`는 이미 생성된 binary dataset(B1~B5)을 읽어서
gray mask와 boundary band를 생성합니다.

실행:

```bash
./scripts/run_graymask_for_binarysets.sh
```

### 출력 결과

각 dataset 폴더 내부에 아래 항목들이 생성됩니다.

* `raw/mask_gray`
* `raw/mask_band`
* `raw/mask_gray_meta`
* `manifest_gray.csv`

### 이 단계의 의미

이 단계는 단순히 binary를 grayscale로 바꾸는 작업이 아닙니다.
**edge band를 생성하고, 그 band 내부에만 grayscale uncertainty를 주입하는 과정**입니다.

즉,

* base binary mask 1장
* → 여러 개의 gray-derived sample

형태로 파생 데이터가 생성됩니다.

---

### Step C. Projector용 1080p Window Generation

`run_project_windows.sh`는 binary 또는 gray mask를
실제 프로젝터에 투사할 수 있는 **1920×1080 window 이미지**로 변환합니다.

실행:

```bash
./scripts/run_project_windows.sh
```

### 출력 결과

* `raw/window_1080p` : binary용 projector window
* `raw/window_1080p_gray` : gray용 projector window

### 참고

현재 rig 설정 기준으로 mask는 1920×1080 black canvas의 특정 위치에 삽입됩니다.
project 위치는 아래 설정 파일에서 수정할 수 있습니다.

```bash
pipeline/configs/rig/rig_default.yaml
```

### 이 단계에서 추가로 수행되는 일

이 task는 단순히 window 이미지만 생성하는 것이 아니라,
이후 LabVIEW/CMOS 실험에서 사용할 폴더 구조도 함께 준비합니다.

예를 들어 아래 폴더들이 함께 준비됩니다.

* `raw/light_distribution`
* `raw/light_distribution_gray`

그리고 batch 단위 폴더 구조도 생성됩니다.

예:

* `window_1080p/batch_0000/...`
* `window_1080p_gray/batch_0000/...`

---

### Step D. Measured Light Distribution (LD) Acquisition

이 단계는 `.sh` 스크립트로 직접 실행하는 단계는 아니며,
실제 실험 장비(LabVIEW + projector + CMOS)를 이용해 수행합니다.

### 수행 방식

1. `window_1080p/batch_xxxx` 또는 `window_1080p_gray/batch_xxxx`에 생성된 이미지를
2. LabVIEW를 통해 순차적으로 projector에 투사하고
3. CMOS 센서로 측정한 light distribution 이미지를 저장합니다.

### 저장 위치

* binary LD:

  * `raw/light_distribution/batch_xxxx`

* gray LD:

  * `raw/light_distribution_gray/batch_xxxx`

즉, 이 단계에서 **projected mask ↔ measured LD** 관계가 실제 실험 데이터로 취득됩니다.

---

### Step E. Pairing

`run_pairing_pipeline.sh`는 실험으로 취득한 raw dataset에서
**window index ↔ LD index ↔ mask filename**을 맞춰서
학습 가능한 형태의 paired dataset으로 정리합니다.

실행:

```bash
./scripts/run_pairing_pipeline.sh
```

### 입력으로 사용하는 항목

기본적으로 다음 파일/폴더를 참조합니다.

#### Binary

* `raw/window_1080p`
* `raw/light_distribution`
* `raw/mask_input`
* `manifest.csv`

#### Gray

* `raw/window_1080p_gray`
* `raw/light_distribution_gray`
* `raw/mask_gray`
* `manifest_gray.csv`

### 출력 결과

* `pairing/binary_mask_128`
* `pairing/binary_rawLD_1600`
* `pairing/gray_mask_128`
* `pairing/gray_rawLD_1600`
* `pairing/*_meta`
* `pairs.csv`
* `pairing_report.json`

### 이 단계의 의미

실험 중에는 batch마다 index가 어긋나거나
파일명이 깔끔하게 정렬되지 않을 수 있습니다.

이 스크립트는 dataset별 rule을 이용해
**정확한 mask-LD pair를 다시 구성하는 역할**을 합니다.

예를 들어:

* **B1**: batch별 offset 규칙이 다를 수 있음
* **B2 ~ B5**: 비교적 단순한 offset 구조

---

### Step F. Registration Parameter Check

`preprocess` 단계에서는 LD 이미지를 그대로 사용하지 않고,
**rotation / scale / translation을 적용한 뒤 crop된 aligned image**를 사용합니다.

따라서 dataset별 registration parameter를 확인해야 합니다.

실행 예시:

```bash
python tools/manual_alignment_pygame_v2.py --dataset /home/swpants05/Desktop/26-1_UROP/raw_datasets/B2_Grid_hf --mode binary
python tools/manual_alignment_pygame_v2.py --dataset /home/swpants05/Desktop/26-1_UROP/raw_datasets/B2_Grid_hf --mode gray
```

### 목적

* binary / gray 각각에 대해
* measured LD와 target mask가 잘 정렬되도록
* rotation / scale / translation 값을 조정

필요 시 dataset 폴더 내부에 `registration_params.json` 등을 저장해 사용할 수 있습니다.

---

### Step G. Preprocess

`run_preprocess_pipeline.sh`는 `pairing/pairs.csv`를 입력으로 받아
최종 학습용 중간 산출물(interim processed data)을 생성합니다.

실행:

```bash
./scripts/run_preprocess_pipeline.sh
```

### 출력 결과

* `mask_128`
* `mask_160`
* `mask_1280`
* `ld_1280_aligned`
* optional threshold images
* `meta/*.json`
* `index.csv`
* `qc.csv`

### 각 항목 설명

* **mask_128**
  기본 binary/gray mask

* **mask_160**
  128 mask에 padding 추가

* **mask_1280**
  160 mask를 8배 upsample

* **ld_1280_aligned**
  registration 적용 후 정렬된 LD 이미지

* **meta/*.json**
  샘플별 메타데이터

* **index.csv**
  최종 샘플 인덱스

* **qc.csv**
  quality control metric

### QC 관련 참고

설정에 따라 다음과 같은 QC metric이 포함될 수 있습니다.

* IoU
* Dice
* NCC

---

### Step H. Train Pack Extraction (Optional)

최종적으로 `tools/extract_trainpack.py`를 사용하면
모델 학습용 train/val/test split이 포함된 pack을 생성할 수 있습니다.

출력 예시:

* `images/{mode}/mask_1280`
* `images/{mode}/ld_1280`
* `thr/...`
* `manifest.csv`
* `splits/train.txt`
* `splits/val.txt`
* `splits/test.txt`
* `dataset_card.json`

---

## 4. Train Pack Extraction Examples

### 4-1. Full Pack

```bash
python tools/extract_trainpack.py \
  --src_root ~/Desktop/26-1_UROP/raw_datasets \
  --datasets all \
  --dst_root ~/Desktop/trainpacks \
  --pack_name TrainPack_allinone_COPY_1280 \
  --modes both \
  --split 0.9,0.05,0.05 \
  --seed 1234 \
  --link_mode copy \
  --include_meta \
  --thr_fixed auto
```

### 4-2. Mini Pack

```bash
python tools/extract_trainpack.py \
  --src_root ~/Desktop/26-1_UROP/raw_datasets \
  --datasets all \
  --dst_root ~/Desktop/trainpacks \
  --pack_name MiniPack_allinone_COPY_1280 \
  --modes both \
  --mini \
  --max_per_dataset 200 \
  --split 0.9,0.05,0.05 \
  --seed 777 \
  --link_mode copy \
  --include_meta \
  --thr_fixed auto
```

### 4-3. Binary-only Pack

```bash
python tools/extract_trainpack.py \
  --src_root ~/Desktop/26-1_UROP/raw_binay_datasets \
  --datasets all \
  --dst_root ~/Desktop/trainpacks \
  --pack_name TrainPack_binary_maskonly_COPY \
  --modes binary \
  --split 0.9,0.05,0.05 \
  --seed 1234 \
  --link_mode copy \
  --include_meta \
  --mask_only \
  --mask_subdir raw/mask_input \
  --no_raw_ld_1600 \
  --no_thr_random \
  --thr_fixed none
```

### 4-4. Mini Binary-only Pack

```bash
python tools/extract_trainpack.py \
  --src_root ~/Desktop/26-1_UROP/raw_binay_datasets \
  --datasets all \
  --dst_root ~/Desktop/trainpacks \
  --pack_name MiniPack_binary_maskonly_COPY \
  --modes binary \
  --split 0.9,0.05,0.05 \
  --seed 1234 \
  --link_mode copy \
  --include_meta \
  --mask_only \
  --mask_subdir raw/mask_input \
  --mini \
  --max_per_dataset 200 \
  --no_raw_ld_1600 \
  --no_thr_random \
  --thr_fixed none
```

---

## 5. Summary

정리하면 전체 과정은 다음과 같습니다.

```text
Binary mask 생성
    ↓
Gray mask / boundary band 생성
    ↓
Projector window 생성
    ↓
LabVIEW + CMOS로 measured LD 취득
    ↓
Mask-LD pairing
    ↓
Registration + preprocess
    ↓
Train pack 추출
```

이 파이프라인을 통해
**synthetic mask 생성부터 실제 optical measurement 기반 paired dataset 구성까지**
일관된 방식으로 처리할 수 있습니다.

```
