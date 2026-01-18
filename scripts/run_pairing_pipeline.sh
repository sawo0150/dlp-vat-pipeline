#!/bin/bash
# ./scripts/run_pairing_pipeline.sh
# ==============================================================================
# DLP Data Pairing Pipeline
# 목적: Raw Dataset(Window/LD)의 파일명 대응 관계를 맞추어 'pairing' 폴더로 정규화
# 전제: 
#   1. configs/pairing/base.yaml, B1.yaml, B2.yaml 파일이 존재해야 함
#   2. raw_datasets 폴더 내에 대상 데이터셋들이 존재해야 함
# ==============================================================================

# 1. 경로 설정
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$PROJECT_ROOT/pipeline/src:$PYTHONPATH"

# 2. 처리할 데이터셋과 사용할 Config 매핑
# 형식: "DatasetName:PairingConfigName"
# 설명:
#   - B1: 복잡한 Offset(-1, +1, 0 혼재) 구조 -> B1.yaml 사용
#   - B2~B5: 모든 Batch에서 Offset이 0인 동일 구조 -> B2.yaml 공통 사용
DATASETS=(
    "B1_Shape_Cutout:B1"
    "B2_Grid_hf:B2"
    "B3_Stripe_phase:B2"
    "B4_mix_Comp:B2"
    "B5_imagenet:B2"
)

echo "=========================================="
echo " Project Root: $PROJECT_ROOT"
echo " Task Type: Pairing (Raw -> Paired)"
echo " Target Datasets: ${#DATASETS[@]} sets"
echo "=========================================="

# 에러 발생 시 즉시 중단
set -e

# 공통 Hydra 플래그
# - task=pair: PairTask 실행
# - dataset.source=raw: Raw 데이터셋 폴더에서 작업 수행
COMMON_FLAGS="task=pair dataset.source=raw hydra.run.dir=. hydra.output_subdir=null"

for ENTRY in "${DATASETS[@]}"; do
    # 문자열 파싱 (DatasetName과 ConfigName 분리)
    # 예: "B1_Shape_Cutout:B1" -> DB_NAME="B1_Shape_Cutout", CONF_NAME="B1"
    IFS=":" read -r DB_NAME CONF_NAME <<< "$ENTRY"

    echo ""
    echo "----------------------------------------------------------------"
    echo " [Processing] $DB_NAME"
    echo " [Config]     configs/pairing/$CONF_NAME.yaml"
    echo "----------------------------------------------------------------"

    # Python 파이프라인 실행
    # dataset.load_id: 대상 데이터셋 폴더명
    # pairing: 사용할 페어링 규칙 파일 이름 (확장자 제외)
    python3 -m dlp_pipeline.main \
        $COMMON_FLAGS \
        dataset.load_id="$DB_NAME" \
        pairing="$CONF_NAME" \
        seed=1234
        
    echo " -> Completed: $DB_NAME"
done

echo ""
echo "=========================================="
echo " All Pairing tasks completed successfully!"
echo " Check 'pairing' folder inside each dataset directory."
echo "=========================================="