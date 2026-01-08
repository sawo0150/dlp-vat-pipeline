#!/usr/bin/bash

# 1. 이 스크립트 파일(run_pipeline.sh)이 있는 경로를 찾습니다.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 2. 프로젝트의 루트 폴더(dlp-vat-pipeline)를 찾습니다. (scripts 폴더의 상위)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 3. PYTHONPATH 환경변수에 'pipeline/src' 경로를 추가합니다.
# 이를 통해 Python이 'src' 폴더 안의 'dlp_pipeline'을 인식하게 됩니다.
export PYTHONPATH="$PROJECT_ROOT/pipeline/src:$PYTHONPATH"

# 4. 모듈 실행 (이제 dlp_pipeline을 찾을 수 있습니다)
# "$@"는 스크립트 실행 시 입력한 인자(예: task=generate)를 그대로 전달합니다.
python3 -m dlp_pipeline.main "$@"