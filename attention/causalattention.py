import torch
import torch.nn as nn

x = torch.tensor(
  [[0.43, 0.15, 0.89],  # Your     (x^1)
   [0.55, 0.87, 0.66],  # journey  (x^2)
   [0.57, 0.85, 0.64],  # starts   (x^3)
   [0.22, 0.58, 0.33],  # with     (x^4)
   [0.77, 0.25, 0.10],  # one      (x^5)
   [0.05, 0.80, 0.55]]  # step     (x^6)
)

d_in = 3
d_out = 2

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.tril(torch.ones(context_length, context_length), diagonal=0))

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        # Calculate attention scores
        attn_scores = queries @ keys.transpose(1, 2)
        
        # Apply causal mask
        num_tokens = x.shape[1]
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        
        # Normalize attention scores
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)
        
        # Compute context vectors
        context_vec = attn_weights @ values
        return context_vec

# Stack batch
batch = torch.stack((x, x), dim=0)
print("Batch shape:", batch.shape)

torch.manual_seed(789)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("Context vectors shape:", context_vecs.shape)
