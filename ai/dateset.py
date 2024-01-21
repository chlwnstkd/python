# dataset.py
from torch.utils.data import Dataset
from tokenizers import Tokenizer

class MyDataset(Dataset):
    def __init__(self, text_path, tokenizer_path, seq_length):
        super().__init__()
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.pad_token = self.tokenizer.encode("<pad>").ids[0]
        self.unk_token = self.tokenizer.encode("<unk>").ids[0]
        self.bos_token = self.tokenizer.encode("<s>").ids[0]
        self.eos_token = self.tokenizer.encode("</s>").ids[0]
        self.mask_token = self.tokenizer.encode("<mask>").ids[0]
        self.input_ids = []
        buffer = []
        with open(text_path, "r") as f:
            for text in f.readlines():
                buffer.extend(self.tokenizer.encode(text).ids)

                # eos, bos 토큰을 붙이기 위해 seq_length-2 만큼 자른다.
                while len(buffer) >= seq_length - 2:
                    input_id = (
                        [self.bos_token] + buffer[: seq_length - 2] + [self.eos_token]
                    )
                    self.input_ids.append(input_id)
                    buffer = buffer[seq_length - 2 :]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx]