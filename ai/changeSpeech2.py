import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np

# 1. 데이터 불러오기
file_path = 'csv/persona/1.tsv'
df = pd.read_csv(file_path, delimiter='\t')

# 2. 데이터 분할
train_data, val_data = train_test_split(df, test_size=0.2, random_state=42)

# 3. 토큰화 및 모델 구성
tokenizers = {}
train_padded = {}
val_padded = {}
models = {}

# 문체 종류 추가
styles = ['formal', 'informal', 'android', 'azae', 'chat', 'choding', 'emoticon', 'enfp', 'gentle', 'halbae',
          'halmae', 'joongding', 'king', 'naruto', 'seonbi', 'sosim', 'translator']

for column in styles:
    # NaN 값을 실제 NaN 값으로 대체
    train_data[column] = train_data[column].astype(str).replace('nan', np.nan).fillna('NaN').copy()
    val_data[column] = val_data[column].astype(str).replace('nan', np.nan).fillna('NaN').copy()

    # 각 문체에 대한 토큰화
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(pd.concat([train_data[column], val_data[column]]))

    train_sequences = tokenizer.texts_to_sequences(train_data[column])
    val_sequences = tokenizer.texts_to_sequences(val_data[column])

    train_padded[column] = pad_sequences(train_sequences)
    val_padded[column] = pad_sequences(val_sequences)

    tokenizers[column] = tokenizer

    # 모델 구성
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizers[column].word_index) + 1, output_dim=64,
                        input_length=train_padded[column].shape[1]))
    model.add(LSTM(128))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    models[column] = model

# 4. 모델 학습
for column in styles:
    # 'NaN' 값을 제외하고 정수로 변환
    train_data[column] = pd.to_numeric(train_data[column], errors='coerce')
    val_data[column] = pd.to_numeric(val_data[column], errors='coerce')

    # NaN 값을 0으로 대체 (또는 다른 값으로 대체)
    train_data[column].fillna(0, inplace=True)
    val_data[column].fillna(0, inplace=True)

    # 정수로 변환
    train_data[column] = train_data[column].astype(int)
    val_data[column] = val_data[column].astype(int)

    models[column].fit(train_padded[column], train_data[column].astype(int), epochs=5,
                       validation_data=(val_padded[column], val_data[column].astype(int)))

# 5. 모델 평가
for column in styles:
    val_loss, val_acc = models[column].evaluate(val_padded[column], val_data[column].astype(int))
    print(f'Validation Accuracy for {column}: {val_acc}')

# 6. 문장 변환 함수
def transform_sentence(input_text, tokenizers, models):
    # 입력값 토큰화
    input_sequences = tokenizers['formal'].texts_to_sequences([input_text])
    input_padded = pad_sequences(input_sequences, maxlen=train_padded['formal'].shape[1])

    # 모델 예측
    predictions = {}
    for column in styles:
        predictions[column] = models[column].predict(input_padded)

    # 예측값을 가장 높은 확률을 가진 문체로 변환
    predicted_style = max(predictions, key=predictions.get)

    return predicted_style

# 모델 저장
for column in styles:
    models[column].save(f'{column}_model.h5')
