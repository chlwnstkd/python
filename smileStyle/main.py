# main.py
from data_preparation import TextStyleTransferDataset, prepare_datasets
from model_definition import create_model, train_model
from inference import generate_text
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from transformers import pipeline
import pandas as pd

# 데이터 로드
df = pd.read_csv("smilestyle_dataset.tsv", sep="\t")

target_styles = df.columns

# 데이터를 훈련 및 테스트 세트로 분할
df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)

# 토크나이저 로드
model_name = "bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained('model_name')

# 데이터셋 준비
train_dataset, test_dataset = prepare_datasets(df_train, df_test, tokenizer)

# 모델 생성 및 훈련
model = create_model(model_name)
model_path = "/content/drive/MyDrive/data/text-transfer-smilegate-bart-eos/"
train_model(model, train_dataset, test_dataset, tokenizer, model_path)

# 기타 메인 로직 추가: 생성된 문장을 특정 스타일로 변환하여 출력
src_text = """
어쩌다 마주친 그대 모습에
내 마음을 빼앗겨 버렸네
어쩌다 마주친 그대 두 눈이
내 마음을 사로잡아 버렸네
그대에게 할 말이 있는데
왜 이리 용기가 없을까
음 말을 하고 싶지만 자신이 없어
내 가슴만 두근두근
답답한 이 내 마음
바람 속에 날려 보내리
"""

nlg_pipeline = pipeline('text-generation', model=model_path, tokenizer=model_name)


print("입력 문장:", src_text)
for style in target_styles:
    print(style, generate_text(nlg_pipeline, src_text, style, num_return_sequences=1, max_length=1000)[0])
