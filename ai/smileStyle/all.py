
# 라이브러리 임포트
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, PreTrainedTokenizerFast
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

# 데이터 불러오기 및 전처리
df = pd.read_csv("smilestyle_dataset.tsv", sep="\t")
display(df.head())
display(df.isna().mean())
display(df.describe())

row_notna_count = df.notna().sum(axis=1)
row_notna_count.plot.hist(bins=row_notna_count.max())
plt.show()

df = df[row_notna_count >= 2]
print(len(df))

# 모델 및 토크나이저 설정
model_name = "gogamza/kobart-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 데이터셋 클래스 정의
class TextStyleTransferDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: PreTrainedTokenizerFast):
        self.df = df
        self.tokenizer = tokenizer

        # 스타일 맵 정의
        self.style_map = {
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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index, :].dropna().sample(2)
        text1 = row[0]
        text2 = row[1]
        target_style = row.index[1]

        target_style_name = self.style_map.get(target_style, target_style)

        encoder_text = f"{target_style_name} 말투로 변환:{text1}"
        decoder_text = f"{text2}{self.tokenizer.eos_token}"
        model_inputs = self.tokenizer(encoder_text, max_length=64, truncation=True)

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(decoder_text, max_length=64, truncation=True)
        model_inputs['labels'] = labels['input_ids']
        del model_inputs['token_type_ids']

        return model_inputs

# 학습 및 테스트 데이터셋 생성
dataset = TextStyleTransferDataset(df, tokenizer)
out = dataset[0]
print(out['input_ids'])
print(out['labels'])
print(tokenizer.decode(out['input_ids']))
print(tokenizer.decode(out['labels']))

# 학습을 위해 train, test set으로 나누기
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)
print(len(df_train), len(df_test))

# 학습을 위한 데이터셋 클래스 생성
train_dataset = TextStyleTransferDataset(df_train, tokenizer)
test_dataset = TextStyleTransferDataset(df_test, tokenizer)

# 모델 및 트레이너 설정
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# 학습을 위한 설정
model_path = "model/1/"
training_args = Seq2SeqTrainingArguments(
    output_dir=model_path,
    overwrite_output_dir=True,
    num_train_epochs=24,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_steps=500,
    save_steps=1000,
    warmup_steps=300,
    prediction_loss_only=True,
    evaluation_strategy="steps",
    save_total_limit=3
)

# 트레이너 생성 및 학습
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

# 모델 저장
trainer.save_model("model/1/")

# 텍스트 생성 파이프라인
from transformers import pipeline

nlg_pipeline = pipeline('text2text-generation', model=model_path, tokenizer=model_name)

# 텍스트 생성 함수 정의
def generate_text(pipe, text, target_style, num_return_sequences=5, max_length=60):
    target_style_name = dataset.style_map.get(target_style, target_style)
    text = f"{target_style_name} 말투로 변환:{text}"
    out = pipe(text, num_return_sequences=num_return_sequences, max_length=max_length)
    return [x['generated_text'] for x in out]

# 예시 텍스트 생성
target_styles = df.columns
src_text = """
어쩌다 마주친 그대 모습에
내 마음을 빼앗겨 버렸네
어쩌다 마주친 그대 두 눈이
내 마음을 사로잡아 버
"""

print("입력 문장:", src_text)
for style in target_styles:
    print(style, generate_text(nlg_pipeline, src_text, style, num_return_sequences=1, max_length=1000)[0])
