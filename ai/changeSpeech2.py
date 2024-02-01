import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Concatenate, Input
from tensorflow.keras import Model
import numpy as np

# CSV 파일 로드
df = pd.read_csv('csv/persona/2.csv')

# 데이터 전처리
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['utterance_text'])
total_words = len(tokenizer.word_index) + 1

# 텍스트 데이터를 토큰화하여 시퀀스로 변환
input_sequences = tokenizer.texts_to_sequences(df['utterance_text'])
input_sequences = pad_sequences(input_sequences, padding='post')

# Label을 만들 때에는 다음 문장을 사용
labels = tokenizer.texts_to_sequences(df['utterance_text'].apply(lambda x: " ".join(x.split()[1:])))
labels = pad_sequences(labels, padding='post')


# profile_gender와 profile_age를 숫자로 매핑
gender_mapping = {'N/A': 0, '여성': 1, '남성': 2}
age_mapping = {'20대': 1, '30대': 2, '40대': 3, '50대': 4, '60대 이상': 5}

df['profile_gender'] = df['profile_gender'].map(gender_mapping)
df['profile_age'] = df['profile_age'].map(age_mapping)

# 입력 데이터와 레이블 생성
X = np.column_stack((df['profile_gender'], df['profile_age']))  # 데이터를 수평으로 쌓음
y = np.argmax(labels, axis=-1)

# 모델 구축
embedding_dim = 100

input_layer = Input(shape=(input_sequences.shape[1],))
embedding_layer = Embedding(input_dim=total_words, output_dim=embedding_dim, input_length=input_sequences.shape[1])(input_layer)
lstm_layer = LSTM(100)(embedding_layer)
output_layer = Dense(total_words, activation='softmax')(lstm_layer)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 훈련
model.fit(input_sequences, y, epochs=10, validation_split=0.2)  # validation_data 대신 validation_split 사용

# 가장 성능이 좋은 모델 저장
model.save('best_model.h5')

# 말투 생성 함수
def generate_response(model, tokenizer, profile_gender, input_text, temperature=1.0, top_k=5):
    input_seq = tokenizer.texts_to_sequences([input_text])
    padded_input = pad_sequences(input_seq, maxlen=input_sequences.shape[1], padding='post')

    # profile_gender를 모델의 입력에 추가
    profile_input = np.array([[profile_gender]])

    # 모델을 사용하여 응답 생성
    predicted_index = np.argmax(model.predict(padded_input), axis=-1)
    predicted_text = tokenizer.index_word.get(predicted_index[0], "")

    return predicted_text

def calculate_similarity(model, tokenizer, text1, text2):
    # 텍스트를 토큰화하여 시퀀스로 변환
    seq1 = tokenizer.texts_to_sequences([text1])[0]
    seq2 = tokenizer.texts_to_sequences([text2])[0]

    # 패딩 적용
    padded_seq1 = pad_sequences([seq1], maxlen=input_sequences.shape[1], padding='post')
    padded_seq2 = pad_sequences([seq2], maxlen=input_sequences.shape[1], padding='post')

    # 임베딩 추출
    embedding1 = model.layers[1](padded_seq1)
    embedding2 = model.layers[1](padded_seq2)

    # 코사인 유사도 계산
    similarity = cosine_similarity(embedding1.numpy(), embedding2.numpy())[0][0]

    return similarity

# 가장 성능이 좋은 모델 로드
best_model = load_model('best_model.h5')

# 예시: 입력값에 대한 말투 생성
input_text = "어제는 좋은 날이었어."
profile_gender = 2  # 남성

similarity_score = calculate_similarity(best_model, tokenizer, input_text1, input_text2)
print(f"두 문장 간의 유사도: {similarity_score}")
generated_response = generate_response(best_model, tokenizer, profile_gender, input_text)
print(f"입력: {input_text}")
print(f"변경된 응답: {generated_response}")
