#!/bin/bash
# ./scripts/run_preprocess_pipeline.sh
# ==============================================================================
# DLP Preprocess Pipeline (Paired -> Processed)
# 목적: pairing/pairs.csv 기반으로 mask 파생물(128/160/1280),
#      LD 정렬+crop(1280), (옵션) threshold/QC/index/meta 생성
# 전제:
#   1. raw_datasets/<DATASET_NAME>/pairing/pairs.csv 가 존재해야 함
#   2. configs/preprocess/standard.yaml 이 존재해야 함
# ==============================================================================

# 1. 경로 설정
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$PROJECT_ROOT/pipeline/src:$PYTHONPATH"

# 2. 처리할 데이터셋 목록
DATASETS=(
    "B1_Shape_Cutout"
    "B2_Grid_hf"
    "B3_Stripe_phase"
    "B4_mix_Comp"
    "B5_imagenet"
)

echo "=========================================="
echo " Project Root: $PROJECT_ROOT"
echo " Task Type: Preprocess (Paired -> Processed)"
echo " Target Datasets: ${#DATASETS[@]} sets"
echo "=========================================="

set -e

# 공통 Hydra 플래그
# - task=preprocess: Preprocessor 실행
# - dataset.source=raw: raw_datasets에서 대상 로드
# - preprocess=standard: preprocess 파라미터 프리셋 선택
COMMON_FLAGS="task=preprocess dataset.source=raw preprocess=standard hydra.run.dir=. hydra.output_subdir=null"

for DB_NAME in "${DATASETS[@]}"; do
    echo ""
    echo "----------------------------------------------------------------"
    echo " [Processing] $DB_NAME"
    echo " [Config]     configs/preprocess/standard.yaml"
    echo "----------------------------------------------------------------"

    python3 -m dlp_pipeline.main \
        $COMMON_FLAGS \
        dataset.load_id="$DB_NAME" \
        seed=1234

    echo " -> Completed: $DB_NAME"
done

echo ""
echo "=========================================="
echo " All Preprocess tasks completed successfully!"
echo " Check 'interim/processed' folder inside each dataset directory."
echo "=========================================="
