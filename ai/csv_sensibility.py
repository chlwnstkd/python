import os
import pandas as pd
from zipfile import ZipFile
import json

# 폴더 경로 설정
folder_path = 'C:\\018.감성대화\\Validation_221115_add\\라벨링데이터'

# 결과를 저장할 CSV 파일 경로
csv_file_path = 'csv/validation_sensibility.csv'

# 각 ZIP 파일을 열어서 JSON 데이터를 추출하여 리스트에 저장
json_data_list = []
for zip_file_name in os.listdir(folder_path):
    if zip_file_name.endswith('.zip'):
        zip_file_path = os.path.join(folder_path, zip_file_name)
        with ZipFile(zip_file_path, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                # JSON 파일만 처리
                if file_name.endswith('.json'):
                    with zip_ref.open(file_name) as json_file:
                        json_data = json.load(json_file)
                        for conversation in json_data:
                            profile = conversation.get('profile', {})
                            talk = conversation.get('talk', {}).get('content', {})
                            if profile and talk:
                                persona_id = profile.get('persona-id', '')
                                talk_id = talk.get('id', {}).get('talk-id', '')
                                terminate = talk.get('HS01', '')  # Assuming 'HS01' corresponds to 'terminate'
                                text = talk.get('SS01', '')  # Assuming 'SS01' corresponds to 'text'

                                json_data_list.append({
                                    'Utterance_ID': talk_id,
                                    'Persona_ID': persona_id,
                                    'Terminate': terminate,
                                    'Text': text
                                })

# JSON 데이터를 CSV 형식으로 변환
pd.DataFrame(json_data_list).to_csv(csv_file_path, index=False)

# CSV 파일로 저장
csv_data = pd.read_csv(csv_file_path)
pd.DataFrame(csv_data).to_csv(csv_file_path, index=False)

# 결과 확인
print(csv_data)