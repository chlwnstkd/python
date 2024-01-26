import os
from zipfile import ZipFile
import pandas as pd
import json

# 폴더 경로 설정
validation_folder_path = '/044.페르소나 대화/01-1.정식개방데이터/validation/02.라벨링데이터'  # 검증 데이터 폴더 경로

# 검증 데이터를 저장할 CSV 파일 경로
validation_csv_file_path = 'validation_data.csv'

# 각 ZIP 파일을 열어서 JSON 데이터를 추출하여 리스트에 저장
validation_json_data_list = []
for zip_file_name in os.listdir(validation_folder_path):
    if zip_file_name.endswith('.zip'):
        zip_file_path = os.path.join(validation_folder_path, zip_file_name)
        with ZipFile(zip_file_path, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                # JSON 파일만 처리
                if file_name.endswith('.json'):
                    with zip_ref.open(file_name) as json_file:
                        json_data = json.load(json_file)
                        validation_json_data_list.extend(json_data['utterances'])

# JSON 데이터를 CSV 형식으로 변환
validation_csv_data = []
for utterance in validation_json_data_list:
    validation_csv_data.append({
        'Utterance_ID': utterance['utterance_id'],
        'Persona_ID': utterance['persona_id'],
        'Terminate': utterance['terminate'],
        'Text': utterance['text']
    })

# CSV 파일로 저장
pd.DataFrame(validation_csv_data).to_csv(validation_csv_file_path, index=False)

# CSV 파일을 데이터프레임으로 읽어오기
validation_combined_df = pd.read_csv(validation_csv_file_path)

# 결과 확인
print(validation_combined_df)
