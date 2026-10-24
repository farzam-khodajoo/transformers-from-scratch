import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def casual_mask(T: int, device=None):
    mask = torch.ones((T, T), dtype=torch.bool, device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask.view(1, 1, T, T)

class MultiHeadSelfAttention(nn.Module):
    
    def __init__(self, d_model: int, n_head: int, dropout: float=0.1):
        super().__init__()
        
        assert d_model % n_head == 0, "d_model should be dividable by n_head"
        
        self.n_head = n_head
        self.d_head = d_model // n_head
        # Query, Key, and Value models all in one Linear model
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.projection = nn.Linear(d_model, d_model, bias=False)
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor):
        assert len(x.shape) == 3, "Input should be 3-dimensional"
        batch_size, sequence_length, embedding_dim = x.shape
        
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, sequence_length, 3, self.n_head, self.d_head)
        # Each sub-model has shape of (B, sequence_length, n_head, d_head)
        query, key, value = qkv.unbind(dim=2)
        
        # swapping dimentions: (B, n_head, sequence_length, d_head)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        adjusted_dropout = self.dropout_rate if self.training else 0.0
        
        attention = F.scaled_dot_product_attention(
            query, key, value,
            is_causal=True,
            dropout_p=adjusted_dropout
        )
        
        attention_reshaped = attention.transpose(1, 2).reshape(batch_size, sequence_length, embedding_dim)
        output = self.dropout(self.projection(attention_reshaped))
        return output