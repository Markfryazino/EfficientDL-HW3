import torch

from transformer import PositionalEncoding

VOCAB_LENGTH = 534191
HIDDEN_DIM = 1024
N_HEADS = 8


# автор задания неправ. Нам нужен не TransformerDecoder (потому что у него внутри кросс-аттеншн и нужны выходы энкодера),
# а TransformerEncoder
class WeirdGPT2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(VOCAB_LENGTH, HIDDEN_DIM)
        self.transformer = torch.nn.TransformerEncoderLayer(HIDDEN_DIM, N_HEADS, batch_first=True)
        self.pos_encoding = PositionalEncoding(HIDDEN_DIM)
    
    def forward(self, x: torch.Tensor):
        mask = torch.tril(torch.ones((x.size(1), x.size(1))), diagonal=0).to(x.device)
        padding_mask = (x == 0).to(x.device)
        x_emb = self.embedding(x)
        x_emb_plus_pos = self.pos_encoding(x_emb.transpose(0, 1)).transpose(0, 1)
        return self.transformer(x_emb_plus_pos, mask, padding_mask)
