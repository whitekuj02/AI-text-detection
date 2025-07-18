# train.py

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
import numpy as np
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, LongformerForSequenceClassification, LongformerTokenizer, DebertaV2ForSequenceClassification
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import torch.nn.functional as F
import os

# --- 1. 설정 및 데이터 로드 ---
CONFIG = {
    "data_base": "../data",
}
train_csv = pd.read_csv(f"{CONFIG['data_base']}/final_aug_train.csv")

# --- 2. 데이터 전처리 및 샘플링 ---
label_0 = train_csv[train_csv['generated'] == 0]
label_1 = train_csv[train_csv['generated'] == 1]
count = min(len(label_0), len(label_1))
sampled_0 = label_0.sample(n=6*count, random_state=42)
sampled_1 = label_1.sample(n=count, random_state=42)
train_csv = pd.concat([sampled_0, sampled_1]).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"✅ 샘플링 완료: 총 {len(train_csv)}개")
print(train_csv["generated"].value_counts())

train_df, val_df = train_test_split(
    train_csv,
    test_size=0.01,
    random_state=42,
    stratify=train_csv['generated']
)

# --- 3. 커스텀 데이터셋 정의 ---
class CustomDataset(Dataset):
    def __init__(self, data_df, tokenizer, mode='train'):
        self.data = data_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['full_text']
        inputs = self.tokenizer(
            text, truncation=True, padding='max_length', max_length=512,
            stride=256, return_overflowing_tokens=True, return_tensors="pt"
        )
        n_segments = inputs["input_ids"].size(0)
        seg_idx = random.randint(0, n_segments - 1)
        item = {k: v[seg_idx] for k, v in inputs.items() if k != "overflow_to_sample_mapping"}
        item.pop("token_type_ids", None)
        if self.mode == 'train':
            item["labels"] = int(row["generated"])
        return item

# --- 4. 모델 및 토크나이저 로드 ---
tokenizer = AutoTokenizer.from_pretrained('vaiv/kobigbird-roberta-large')
train_dataset = CustomDataset(train_df, tokenizer, mode='train')
val_dataset = CustomDataset(val_df, tokenizer, mode='train')

device = "cuda" if torch.cuda.is_available() else "cpu"
base_model = AutoModelForSequenceClassification.from_pretrained(
    "vaiv/kobigbird-roberta-large",
    num_labels=2
)
base_model = prepare_model_for_kbit_training(base_model)
lora_config = LoraConfig(
    r=8, lora_alpha=32, target_modules=["query", "value"],
    lora_dropout=0.1, bias="none", task_type="SEQ_CLS"
)
model = get_peft_model(base_model, lora_config)

# --- 5. Trainer 및 훈련 설정 ---
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall, 'auroc': auc}

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        class_weights = torch.tensor([1.0, 6.0], device=logits.device)
        example_weights = class_weights[labels]
        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        # 이 부분은 WeightedTrainer를 사용하신다면 원래 로직을 따르셔도 됩니다.
        # 위 코드는 더 간결한 가중치 적용 방식입니다.
        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir="./",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    max_steps=300000,
    learning_rate=5e-5,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=10000,
    logging_dir="./",
    logging_steps=10000,
    fp16=True,
    report_to="none"
)

trainer = WeightedTrainer(
    model=model, args=training_args, train_dataset=train_dataset,
    eval_dataset=val_dataset, tokenizer=tokenizer, compute_metrics=compute_metrics,
)

# --- 6. 훈련 시작! ---
print("✅ 모델 훈련을 시작합니다.")
trainer.train()
print("✅ 모델 훈련이 완료되었습니다.")