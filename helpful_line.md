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