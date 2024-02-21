# -*- coding: utf-8 -*-


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import pandas as pd


# 모델 및 토크나이저 설정
model_name = "gogamza/kobart-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_path = "/content/drive2/MyDrive/data/text-transfer-smilegate-bart-eos/"  # 저장된 모델 경로

# 모델 로드
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# 텍스트 생성 파이프라인
nlg_pipeline = pipeline('text2text-generation', model=model, tokenizer=model_name)

# 텍스트 생성 함수 정의
def generate_text(pipe, text, target_style, dataset, num_return_sequences, max_length):
    target_style_name = dataset.style_map.get(target_style, target_style)
    formatted_text = "{} 말투로 변환: {}".format(target_style_name, text)
    out = pipe(formatted_text, num_return_sequences=num_return_sequences, max_length=max_length)
    return [x['generated_text'] for x in out]




# 예시 텍스트 생성을 위한 가상의 데이터셋 정의
class VirtualDataset:
    def __init__(self):
        self.style_map = {
            'formal': '문어체',
            'informal': '구어체',
            'android': '안드로이드',
            'azae': '아재',
            'chat': '채팅',
            'choding': '초등학생',
            # 'emoticon': '이모티콘',  애매
            'enfp': '활발한',
            'gentle': '신사',
            'halbae': '할아버지',
            'halmae': '할머니',
            'joongding': '중학생',
            'king': '왕',
            # 'naruto': '나루토', 애매
            'seonbi': '선비',
            'sosim': '소심한',
            'translator': '번역기'
        }

def result(src_text, translator_style):
    dataset = VirtualDataset()
    translation_result = generate_text(nlg_pipeline, src_text, translator_style, dataset, num_return_sequences=1, max_length=1000)[0]
    return translation_result

# 가상의 데이터셋 생성
virtual_df = pd.DataFrame({'col1': ['text1', 'text2']})
dataset = VirtualDataset()

# 예시 텍스트 생성
target_styles = virtual_df.columns
src_text = """
세빈이 누나 바보
"""

def result(src_text, translator_style, nlg_pipeline, dataset):
    translation_result = generate_text(nlg_pipeline, src_text, translator_style, dataset, num_return_sequences=1, max_length=1000)[0]
    return translation_result

print("입력 문장:", src_text)

# 번역기 스타일 선택
translator_style = 'naruto'
translation_result = generate_text(nlg_pipeline, src_text, translator_style, dataset, num_return_sequences=1, max_length=1000)[0]


print(f"{translator_style} 스타일로 변환:", translation_result)

