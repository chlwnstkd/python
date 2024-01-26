# model.py
import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor):
        seq_len = input_ids.size(1)
        position_ids = (
            torch.arange(seq_len, dtype=torch.long).unsqueeze(0).to(input_ids.device)
        )
        position_embeddings = self.position_embeddings(position_ids)
        word_embeddings = self.word_embeddings(input_ids)
        embeddings = word_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings