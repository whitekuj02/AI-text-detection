# predict.py

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os

# --- 1. 설정 및 데이터 로드 ---
CONFIG = {
    "data_base": "../data",
}
test_csv = pd.read_csv(f"{CONFIG['data_base']}/test.csv")

# --- 2. 커스텀 데이터셋 정의 (추론에도 필요) ---
class CustomDataset(Dataset):
    def __init__(self, data_df, tokenizer, mode='eval'): # mode 기본값을 'eval'로
        self.data = data_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['paragraph_text']
        inputs = self.tokenizer(
            text, truncation=True, padding='max_length', max_length=512,
            stride=256, return_overflowing_tokens=True, return_tensors="pt"
        )
        n_segments = inputs["input_ids"].size(0)
        seg_idx = random.randint(0, n_segments - 1)
        item = {k: v[seg_idx] for k, v in inputs.items() if k != "overflow_to_sample_mapping"}
        item.pop("token_type_ids", None)
        return item

# --- 3. 토크나이저 및 모델 로드 ---
# 훈련 시 사용했던 토크나이저와 동일해야 함
tokenizer = AutoTokenizer.from_pretrained('vaiv/kobigbird-roberta-large')
test_dataset = CustomDataset(test_csv, tokenizer, mode='eval')

# 훈련된 모델 체크포인트 불러오기
checkpoint_path = "./checkpoint-280000"
print(f"✅ {checkpoint_path}에서 모델을 불러옵니다.")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# --- 4. 추론 루프 ---
all_probs = []
valid_keys = {"input_ids", "attention_mask"}

print("✅ 추론을 시작합니다.")
with torch.no_grad():
    for i in tqdm(range(len(test_dataset)), desc="Running Inference"):
        batch = test_dataset[i]
        inputs = {k: v.unsqueeze(0).to(model.device) for k, v in batch.items() if k in valid_keys}
        
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        all_probs.extend(probs.cpu().tolist())

# --- 5. 제출 파일 생성 ---
sample_submission = pd.read_csv(f"{CONFIG['data_base']}/sample_submission.csv")
# 'AI가 생성한 글'일 확률은 두 번째 컬럼(인덱스 1)
all_AI_probs = [p[1] for p in all_probs]
sample_submission['generated'] = all_AI_probs
sample_submission.to_csv(f"submit.csv", index=False)

print("✅ 제출 파일 생성이 완료되었습니다.")