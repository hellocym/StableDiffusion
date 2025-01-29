import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embed: int, n_tokens: int):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_embed)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embed))

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        # (Batch_size, Seq_Len) -> (Batch_size, Seq_Len, Dim)
        x = self.token_embedding(tokens)
        # (Batch_size, Seq_Len, Dim)
        x += self.position_embedding
        
        return x
    
class CLIPLayer(nn.Module):
    def __init__(self, n_heads: int, n_embed: int):
        super().__init__()
        self.attention = SelfAttention(n_heads, n_embed)
        self.norm1 = nn.LayerNorm(n_embed)
        self.norm2 = nn.LayerNorm(n_embed)
        self.linear1 = nn.Linear(n_embed, n_embed * 4)
        self.linear2 = nn.Linear(n_embed * 4, n_embed)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # x: (Batch_size, Seq_Len, Dim)
        residue = x

        x = self.norm1(x)
        x = self.attention(x, casual_mask=True)
        x += residue

        residue = x

        x = self.norm2(x)
        x = self.linear1(x)
        # quickGELU
        x = x * torch.sigmoid(1.702 * x)
        x = self.linear2(x)
        x += residue

        return x

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.Module([
            CLIPLayer(12, 768) for i in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)
        # (Batch_size, Seq_Len) -> (Batch_size, Seq_Len, Dim)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        # (Batch_size, Seq_Len, Dim)
        output = self.layernorm(state)

        return output
