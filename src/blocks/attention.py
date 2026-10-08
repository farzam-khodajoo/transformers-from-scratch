import torch
import torch.nn as nn

class SingleHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads, dropout=0.1, bias=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        assert embedding_dim % n_heads == 0, "embedding_dim must be divisible by n_heads"
        
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads
        
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=bias, dropout=dropout)
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=bias, dropout=dropout)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=bias, dropout=dropout)
        
        self.projection = nn.Linear(embedding_dim, embedding_dim, bias=bias, dropout=dropout)
        
    def forward(self, x):
        # Batch, Time sequence, Channels (features)
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        scores = Q @ K.transpose(-2, -1) / (self.embedding_dim ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)
        output = attention_weights @ V
        
        return output, attention_weights


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)