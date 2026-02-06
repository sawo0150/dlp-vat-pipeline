가상환경 활성화

기존에 만들다 만 게 있다면 삭제
conda env remove -n dlp -y

1. 빈 환경 생성 (Python 3.10 버전 지정) - 이건 금방 끝납니다.
- conda create -n dlp python=3.10 -y

2. 활성화
- conda activate dlp

3. 방금 만든 requirements.txt 설치
- pip install -r requirements.txt

4. 프로젝트 루트(dlp-vat-pipeline 폴더)에서 실행
pip install -e ./common
pip install -e ./ml
pip install -e ./pipeline

conda activate dlp
conda deactivate

비활성화
- deactivate

(중요) 한 venv에서 “pipeline+ml+common” editable 설치하기
source .venv/bin/activate
pip install -e ./common
pip install -e ./pipeline
pip install -e ./ml

./scripts/run_pipeline.sh

python manual_alignment_pygame_v2.py --dataset /home/swpants05/Desktop/26-1_UROP/raw_datasets/B2_Grid_hf --mode binary

2) 실행 명령어 예시
- full pack
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

- minipack
python tools/extract_trainpack.py \
  --src_root ~/Desktop/26-1_UROP/raw_datasets \
  --datasets all \
  --dst_root ~/Desktop/trainpacks \
  --pack_name MiniPack_allinone_COPY_1280 \
  --modes both \
  --mini --max_per_dataset 200 \
  --split 0.9,0.05,0.05 \
  --seed 777 \
  --link_mode copy \
  --include_meta \
  --thr_fixed auto

- binarypack
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
