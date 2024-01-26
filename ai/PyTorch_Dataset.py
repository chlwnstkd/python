from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):
    def __init__(self, tokenized_inputs, labels):
        self.tokenized_inputs = tokenized_inputs
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.tokenized_inputs['input_ids'][idx],
            'attention_mask': self.tokenized_inputs['attention_mask'][idx],
            'label': self.labels[idx]
        }