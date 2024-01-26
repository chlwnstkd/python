from transformers import BertForSequenceClassification, BertTokenizer

# 불러올 모델과 토크나이저의 경로
model_path = 'path/to/save/model'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 모델 불러오기
model = BertForSequenceClassification.from_pretrained(model_path)

# 예측할 문장
text = "Your input text goes here."

# 문장을 토큰화하고 모델에 입력으로 넣기
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)

# 로짓 (logits) 추출
logits = outputs.logits

# 예측 결과 얻기
predicted_class = logits.argmax().item()

# 결과 출력
print(f"Predicted class: {predicted_class}")