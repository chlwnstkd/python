# data_preparation.py
from inference import tokenizer
from tokenizers import Tokenizer
from torch.utils.data import Dataset
import pandas as pd
from inference import style_map

class TextStyleTransferDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: Tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index, :].dropna().sample(2)
        text1 = row[0]
        text2 = row[1]
        target_style = row.index[1]
        target_style_name = style_map[target_style]

        encoder_text = f"{target_style_name} 말투로 변환:{text1}"
        decoder_text = f"{text2}{self.tokenizer.eos_token}"
        model_inputs = self.tokenizer(encoder_text, max_length=64, truncation=True)

        with self.tokenizer.as_target_tokenizer():
            labels = tokenizer(decoder_text, max_length=64, truncation=True)
        model_inputs['labels'] = labels['input_ids']
        del model_inputs['token_type_ids']

        return model_inputs

def prepare_datasets(train_df, test_df, tokenizer):
    train_dataset = TextStyleTransferDataset(train_df, tokenizer)
    test_dataset = TextStyleTransferDataset(test_df, tokenizer)
    return train_dataset, test_dataset


