import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torch import nn
from transformers import BertForSequenceClassification
import pandas as pd
import time
from tqdm import tqdm
from PyTorch_Dataset import CustomDataset
from sklearn.preprocessing import LabelEncoder

train_df = pd.read_csv('csv/tranning_sensibility.csv')
val_df = pd.read_csv('csv/validation_sensibility.csv')

# 훈련 데이터셋 문장과 레이블 추출
train_sentences = train_df['Text'].tolist()
train_labels = train_df['Terminate'].tolist()

# 검증 데이터셋 문장과 레이블 추출
val_sentences = val_df['Text'].tolist()
val_labels = val_df['Terminate'].tolist()

# 훈련 데이터셋과 검증 데이터셋을 하나로 합침
all_sentences = train_sentences + val_sentences
all_labels = train_labels + val_labels

# 레이블을 숫자로 인코딩
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(all_labels)

# BERT 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# 전체 데이터셋 문장을 토큰화 및 패딩
tokenized_inputs = tokenizer(all_sentences, padding=True, truncation=True, max_length=256, return_tensors='pt')

# 훈련 데이터셋 및 검증 데이터셋으로 나눔
train_size = len(train_sentences)
train_inputs = {key: val[:train_size] for key, val in tokenized_inputs.items()}
val_inputs = {key: val[train_size:] for key, val in tokenized_inputs.items()}

# 훈련 Dataset 및 DataLoader 초기화
train_dataset = CustomDataset(train_inputs, encoded_labels[:train_size])
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# 검증 Dataset 및 DataLoader 초기화
val_dataset = CustomDataset(val_inputs, encoded_labels[train_size:])
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# BERT 모델 로드
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=len(label_encoder.classes_))

# 옵티마이저 및 손실 함수 설정
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

num_epochs = 1
total_batches = len(train_dataloader)
start_time = time.time()

for epoch in range(num_epochs):

    # tqdm을 사용하여 간단한 프로그레스 바 출력
    for batch_idx, batch in enumerate(tqdm(train_dataloader, total=total_batches)):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()

    # 현재 epoch이 끝날 때마다 소요된 시간 출력
    elapsed_time = time.time() - start_time
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")

print("Training Finished.")

# 학습된 모델을 저장
model.save_pretrained('model/sensibility')

model.eval()  # 모델을 평가 모드로 변경

val_loss = 0.0
correct_predictions = 0

with torch.no_grad():
    for batch in val_dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)

        val_loss += loss.item()

        # 정확한 예측 수 계산
        predictions = torch.argmax(outputs.logits, dim=1)
        correct_predictions += torch.sum(predictions == labels).item()

# 검증 데이터셋에 대한 평균 손실 및 정확도 계산
avg_val_loss = val_loss / len(val_dataloader)
accuracy = correct_predictions / len(val_dataset)

print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy * 100:.2f}%')
