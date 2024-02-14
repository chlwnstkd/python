import json
import os
import zipfile
import pandas as pd

# JSON 파일을 저장할 빈 리스트
all_data = []

# 폴더 내의 모든 zip 파일에 대해 반복
folder_path = "/044.페르소나 대화/01-1.정식개방데이터/validation/02.라벨링데이터"

for file_name in os.listdir(folder_path):
    if file_name.endswith(".zip"):
        file_path = os.path.join(folder_path, file_name)

        # zip 파일 압축 해제
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(folder_path)

        # 압축 해제된 JSON 파일 읽기
        json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]

        for json_file in json_files:
            json_file_path = os.path.join(folder_path, json_file)

            # JSON 파일 읽기
            with open(json_file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                all_data.append(data)

            # 읽은 후에 JSON 파일 삭제 (필요에 따라 삭제 여부를 결정할 수 있습니다)
            os.remove(json_file_path)

# 전체 데이터에 대한 전처리 수행
rows = []

for data in all_data:
    personas = data["info"]["personas"]
    utterances = data["utterances"]

    for persona in personas:
        for utterance in utterances:
            if utterance["persona_id"] == persona["persona_id"]:
                row = {
                    "persona_id": persona["persona_id"],
                    "avg_rating": persona["evaluation"]["avg_rating"],
                    "profile_gender": next(
                        (p["profile_minor"] for p in persona["persona"] if p["profile_major"] == "성별"), "N/A"),
                    "profile_age": next((p["profile_minor"] for p in persona["persona"] if p["profile_major"] == "연령대"),
                                        "N/A"),
                    "utterance_text": utterance["text"]
                }
                rows.append(row)

# 데이터프레임 생성
df = pd.DataFrame(rows)

# CSV 파일로 저장
df.to_csv("csv/persona/2.csv", index=False, encoding="utf-8")
