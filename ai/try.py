import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. 데이터 불러오기
file_path = 'csv/persona/1.tsv'
df = pd.read_csv(file_path, delimiter='\t')

# 2. 모델 및 토큰화기 초기화
models = {}
tokenizers = {}


# 모델 로드 및 토큰화기 초기화
def load_and_initialize_model(model_name, tokenizer_name):
    model = load_model(f'{model_name}_model.h5')
    tokenizer = Tokenizer()
    # Tokenizer 초기화 및 필요한 설정 수행
    tokenizer.fit_on_texts(df[tokenizer_name].astype(str))

    models[model_name] = model
    tokenizers[tokenizer_name] = tokenizer


# 모델 및 토큰화기 초기화 수행
load_and_initialize_model('android', 'android')
load_and_initialize_model('azae', 'azae')


# 나머지 모델들에 대해서도 동일하게 수행

# 3. 문장 변환 함수
def transform_sentence(input_text):
    predicted_styles = {}

    # 각 모델에 대해 예측 수행
    for model_name, model in models.items():
        tokenizer = tokenizers[model_name]

        # 입력값 토큰화
        input_sequences = tokenizer.texts_to_sequences([input_text])
        input_padded = pad_sequences(input_sequences, maxlen=model.input_shape[1])

        # 모델 예측
        prob = model.predict(input_padded)

        # 예측값을 문체로 변환
        predicted_style = "formal" if prob > 0.5 else "informal"
        predicted_styles[model_name] = predicted_style

    return predicted_styles


# 4. 문장 변환 예시
input_text = "여기에 입력 문장을 넣어주세요."
predicted_styles = transform_sentence(input_text)

# 예측 결과 출력
print(f"입력 문장: {input_text}")
for model_name, predicted_style in predicted_styles.items():
    print(f"{model_name} 예측 결과: {predicted_style}")
