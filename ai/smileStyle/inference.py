# inference.py
from transformers import pipeline
from transformers import BertTokenizer

def generate_text(pipe, text, target_style, num_return_sequences=5):
    target_style_name = style_map[target_style]
    text = f"{target_style_name} 말투로 변환:{text}"
    out = pipe(text, num_return_sequences=num_return_sequences,
               pad_token_id=tokenizer.eos_token_id,
               max_length=100)
    return [filter_generated_result(x['generated_text']) for x in out]

style_map = {
    'formal': '문어체',
    'informal': '구어체',
    'android': '안드로이드',
    'azae': '아재',
    'chat': '채팅',
    'choding': '초등학생',
    'emoticon': '이모티콘',
    'enfp': 'enfp',
    'gentle': '신사',
    'halbae': '할아버지',
    'halmae': '할머니',
    'joongding': '중학생',
    'king': '왕',
    'naruto': '나루토',
    'seonbi': '선비',
    'sosim': '소심한',
    'translator': '번역기'
}

tokenizer = BertTokenizer.from_pretrained('beomi/kcgpt2')

def filter_generated_result(text):
    return text.split("[SEP]")[0]