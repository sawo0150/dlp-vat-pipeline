import os
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from huggingface_hub import login  # 이 줄 추가

# 여기에 아까 발급받은 토큰(hf_...)을 문자열로 넣으세요
# login(token="xxxxxxxx")

# 저장할 경로 설정
OUTPUT_DIR = "../processed_imagenet"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# 1. ImageNet 데이터셋 스트리밍 모드로 불러오기
# 수정됨: use_auth_token=True 삭제
# 추가됨: trust_remote_code=True (혹시 모를 스크립트 실행 권한 문제를 위해 추가하는 것이 좋습니다)
print("Loading dataset stream...")
dataset = load_dataset("imagenet-1k", split="train", streaming=True, trust_remote_code=True)

# 원하는 데이터 수량 설정
TOTAL_IMAGES_TO_PROCESS = 100000 

print(f"Start processing {TOTAL_IMAGES_TO_PROCESS} images...")

count = 0
# 스트리밍 모드는 전체 길이를 모르기 때문에 tqdm에 total을 명시하면 진행바가 예쁘게 나옵니다.
for data in tqdm(dataset, total=TOTAL_IMAGES_TO_PROCESS):
    try:
        # 2. 이미지 추출 및 전처리
        img = data['image'] # 원본 이미지 (PIL 객체)
        label = data['label']
        
        # 128x128 리사이즈 & Grayscale 변환 ('L' mode)
        img_resized = img.resize((128, 128)).convert('L')
        
        # 3. 저장
        # 이미지 파일명 충돌 방지를 위해 count를 앞에 붙임
        save_path = os.path.join(OUTPUT_DIR, f"{count}_{label}.png")
        img_resized.save(save_path)
        
        count += 1
        if count >= TOTAL_IMAGES_TO_PROCESS:
            break
            
    except Exception as e:
        print(f"Error processing image {count}: {e}")
        continue

print(f"Done! Saved {count} images to {OUTPUT_DIR}")