import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias: bool = True, out_proj_bias: bool = True):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, casual_mask=False):
        # x: (Batch_size, Seq_Len, Dim)
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # (Batch_size, Seq_Len, Dim) -> 3 * (Batch_size, Seq_Len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        # (Batch_size, Seq_Len, Dim) -> (Batch_size, Seq_Len, Heads, Dim // Heads) -> (Batch_size, Heads, Seq_Len, Dim // Heads)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (Batch_size, Heads, Seq_Len, Dim // Heads) @ (Batch_size, Heads, Dim // Heads, Seq_Len) -> (Batch_size, Heads, Seq_Len, Seq_Len)
        weight = q @ k.transpose(-1, -2)

        if casual_mask:
            # Mask out the upper triangular part of the matrix
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        # (Batch_size, Heads, Seq_Len, Seq_Len) @ (Batch_size, Heads, Seq_Len, Dim // Heads) -> (Batch_size, Heads, Seq_Len, Dim // Heads)
        output = weight @ v
        # (Batch_size, Heads, Seq_Len, Dim // Heads) -> (Batch_size, Seq_Len, Heads, Dim // Heads)
        output = output.transpose(1, 2)
        output = output.reshape(input_shape)
        output = self.out_proj(output)
        # (Batch_size, Seq_Len, Dim)
        return output
    

class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias: bool = True, out_proj_bias: bool = True):
        super().__init__()

        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        # x: (Batch_size, Seq_Len_Q, Dim_Q)
        # context: (Batch_size, Seq_Len_KV, Dim_KV) = (Batch_size, 77, 768)
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)

        # (Batch_size, Seq_Len, Dim) -> (Batch_size, Seq_Len, Heads, Dim // Heads) -> (Batch_size, Heads, Seq_Len, Dim // Heads)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (Batch_size, Heads, Seq_Len, Dim // Heads) @ (Batch_size, Heads, Dim // Heads, Seq_Len) -> (Batch_size, Heads, Seq_Len, Seq_Len)
        weight = q @ k.transpose(-1, -2)

        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        # (Batch_size, Heads, Seq_Len, Seq_Len) @ (Batch_size, Heads, Seq_Len, Dim // Heads) -> (Batch_size, Heads, Seq_Len, Dim // Heads)
        output = weight @ v
        # (Batch_size, Heads, Seq_Len, Dim // Heads) -> (Batch_size, Seq_Len, Heads, Dim // Heads)
        output = output.transpose(1, 2).contiguous()

        output = output.view(input_shape)

        output = self.out_proj(output)

        return output