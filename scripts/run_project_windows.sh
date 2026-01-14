#!/bin/bash
# ./scripts/run_project_windows.sh
# ==============================================================================
# DLP Project Window Generation Pipeline
# 목적: 이미 생성된 Binary mask / Gray mask로부터 1080p projector window를 일괄 생성
# 전제:
#   1) ./scripts/run_binarydataset_gen.sh 로 binary mask 생성 완료
#   2) ./scripts/run_graymask_gen.sh 로 gray mask 생성 완료 (gray까지 만들 거면)
# ==============================================================================
set -e

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

# 3. 어떤 window를 만들지 선택: both | binary | gray
# 사용 예:
#   ./scripts/run_project_windows.sh both
#   ./scripts/run_project_windows.sh binary
#   ./scripts/run_project_windows.sh gray
WHICH="${1:-both}"

# 4. overwrite 설정: true면 기존 파일도 다시 생성
# 사용 예:
#   OVERWRITE=true ./scripts/run_project_windows.sh both
OVERWRITE="${OVERWRITE:-true}"

echo "=========================================="
echo " Project Root: $PROJECT_ROOT"
echo " Target Datasets: ${#DATASETS[@]} sets"
echo " Project Which: $WHICH"
echo " Overwrite: $OVERWRITE"
echo "=========================================="

# 공통 Hydra 플래그
COMMON_FLAGS="task=project task.which=$WHICH task.overwrite=$OVERWRITE hydra.run.dir=. hydra.output_subdir=null"

for DB_NAME in "${DATASETS[@]}"; do
  echo ""
  echo "----------------------------------------------------------------"
  echo " [Project] $DB_NAME (which=$WHICH, overwrite=$OVERWRITE)"
  echo "----------------------------------------------------------------"

  python3 -m dlp_pipeline.main \
    $COMMON_FLAGS \
    dataset.load_id="$DB_NAME"
done

echo ""
echo "=========================================="
echo " All Project tasks completed successfully!"
echo " Check each dataset folder:"
echo "  - raw/window_1080p/        (binary)"
echo "  - raw/window_1080p_gray/   (gray)"
echo "=========================================="
