#!/bin/bash
# ./scripts/run_binarydataset_gen.sh
# ==============================================================================
# DLP Dataset Generation Pipeline
# 목적: 5가지 데이터셋(B1~B5)을 지정된 비율(SCALE)에 맞춰 일괄 생성
# ==============================================================================

# 1. 경로 설정 (run_pipeline.sh와 동일한 로직)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$PROJECT_ROOT/pipeline/src:$PYTHONPATH"

# 2. SCALE 설정 (기본값: 1000 -> 총 30k 생성)
# 실행 시 인자로 전달 가능 예: ./run_dataset_gen.sh 1000
if [ -z "$1" ]; then
    SCALE=1000
else
    SCALE=$1
fi

echo "=========================================="
echo " Project Root: $PROJECT_ROOT"
echo " Base Scale: $SCALE"
echo "=========================================="

# 에러 발생 시 즉시 중단
set -e

# 공통 Hydra 플래그
# - task=generate: 생성 모드 강제
# - hydra.run.dir=.: 실행 위치에 로그/산출물 생성
COMMON_FLAGS="task=generate dataset.load_id=null hydra.run.dir=. hydra.output_subdir=null"

# ------------------------------------------------------------------------------
# [1] B1_Shape_Cutout (비율: 10.0 * SCALE) -> 예: 10,000
# ------------------------------------------------------------------------------
COUNT=$(( 10 * SCALE ))
echo "[1/5] Generating B1_Shape_Cutout ($COUNT images)..."

python3 -m dlp_pipeline.main \
    $COMMON_FLAGS \
    generator=b1_shape \
    dataset.name="B1_Shape_Cutout" \
    dataset.id_strategy="manual" \
    task.num_images=$COUNT \
    seed=1001

# ------------------------------------------------------------------------------
# [2] B2_Grid_hf (비율: 5.0 * SCALE) -> 예: 5,000
# ------------------------------------------------------------------------------
COUNT=$(( 5 * SCALE ))
echo "[2/5] Generating B2_Grid_hf ($COUNT images)..."

python3 -m dlp_pipeline.main \
    $COMMON_FLAGS \
    generator=b2_grid \
    dataset.name="B2_Grid_hf" \
    dataset.id_strategy="manual" \
    task.num_images=$COUNT \
    seed=2002

# ------------------------------------------------------------------------------
# [3] B3_Stripe_phase (비율: 5.0 * SCALE) -> 예: 5,000
# ------------------------------------------------------------------------------
COUNT=$(( 5 * SCALE ))
echo "[3/5] Generating B3_Stripe_phase ($COUNT images)..."

python3 -m dlp_pipeline.main \
    $COMMON_FLAGS \
    generator=b3_stripe \
    dataset.name="B3_Stripe_phase" \
    dataset.id_strategy="manual" \
    task.num_images=$COUNT \
    seed=3003

# ------------------------------------------------------------------------------
# [4] B4_mix_Comp_hard (비율: 5.0 * SCALE) -> 예: 5,000
# ------------------------------------------------------------------------------
COUNT=$(( 5 * SCALE ))
echo "[4/5] Generating B4_mix_Comp_hard ($COUNT images)..."

python3 -m dlp_pipeline.main \
    $COMMON_FLAGS \
    generator=b4_mix \
    dataset.name="B4_mix_Comp" \
    dataset.id_strategy="manual" \
    task.num_images=$COUNT \
    seed=4004

# ------------------------------------------------------------------------------
# [5] B5_imagenet (비율: 5.0 * SCALE) -> 예: 5,000
# ------------------------------------------------------------------------------
COUNT=$(( 5 * SCALE ))
echo "[5/5] Generating B5_imagenet ($COUNT images)..."

python3 -m dlp_pipeline.main \
    $COMMON_FLAGS \
    generator=b5_imagenet \
    dataset.name="B5_imagenet" \
    dataset.id_strategy="manual" \
    task.num_images=$COUNT \
    seed=5005

echo "=========================================="
echo " All datasets generated successfully!"
echo " Check your 'dataset_root' folder."
echo "=========================================="