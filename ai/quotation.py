import pandas as pd

# CSV 파일 경로
csv_file_path = "csv/persona/2.csv"

# CSV 파일 읽기
df = pd.read_csv(csv_file_path)

# utterance_text 열에 큰따옴표 추가
df['utterance_text'] = df['utterance_text'].apply(lambda x: f'"{x}"')

# 수정된 데이터프레임을 CSV 파일로 저장
df.to_csv(csv_file_path, index=False)
