#!/bin/bash
# ==============================================================================
# DLP GrayMask Generation Pipeline
# 목적: 사전에 생성된 B1~B5 Binary 데이터셋에 대해 GrayMask/Sobolev 변환 수행
# 전제: ./run_binarydataset_gen.sh 가 먼저 실행되어 있어야 함
# ==============================================================================

# 1. 경로 설정
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$PROJECT_ROOT/pipeline/src:$PYTHONPATH"

# 2. 처리할 데이터셋 목록 (dataset.name과 정확히 일치해야 함)
# run_binarydataset_gen.sh에서 정의한 이름들입니다.
DATASETS=(
    "B1_Shape_Cutout"
    "B2_Grid_hf"
    "B3_Stripe_phase"
    "B4_mix_Comp"
    "B5_imagenet"
)

echo "=========================================="
echo " Project Root: $PROJECT_ROOT"
echo " Target Datasets: ${#DATASETS[@]} sets"
echo "=========================================="

# 에러 발생 시 즉시 중단
set -e

# 공통 Hydra 플래그
# - task=graymask: GrayMaskTask 실행
# - task.overwrite=true: 이미 결과가 있어도 덮어쓰기 (필요 시 false로 변경)
# - task.source=manifest: 기존 manifest.csv를 읽어서 처리
COMMON_FLAGS="task=graymask task.overwrite=true task.source=manifest hydra.run.dir=. hydra.output_subdir=null"

for DB_NAME in "${DATASETS[@]}"; do
    echo ""
    echo "----------------------------------------------------------------"
    echo " [Processing] $DB_NAME ..."
    echo "----------------------------------------------------------------"

    # Python 파이프라인 실행
    # dataset.load_id에 폴더명(DB_NAME)을 전달하여 해당 폴더 내부에서 작업 수행
    python3 -m dlp_pipeline.main \
        $COMMON_FLAGS \
        dataset.load_id="$DB_NAME" \
        seed=9999 \
        # seed는 graymask 생성 시 random noise 패턴에 영향을 줍니다.
        # 필요하다면 DB마다 다른 시드를 주거나, 고정할 수 있습니다.
done

echo ""
echo "=========================================="
echo " All GrayMask tasks completed successfully!"
echo " Check 'mask_gray', 'mask_band', 'window_gray' in each dataset folder."
echo "=========================================="