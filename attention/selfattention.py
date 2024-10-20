import torch
import torch.nn as nn
x = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec
d_in=3
d_out=2
torch.manual_seed(789)
sa_v1 = SelfAttention_v1(d_in, d_out)
# print(sa_v1(x))



#Output
# tensor([[0.2996, 0.8053],
#         [0.3061, 0.8210],
#         [0.3058, 0.8203],
#         [0.2948, 0.7939],
#         [0.2927, 0.7891],
#         [0.2990, 0.8040]], grad_fn=<MmBackward0>)

# Using nn.Linear
attention_weights=torch.tensor([])
attention_scores=torch.tensor([])
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out)
        self.W_key   = nn.Linear(d_in, d_out)
        self.W_value = nn.Linear(d_in, d_out)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T # omega
        global attention_scores
        attention_scores = attn_scores
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        global attention_weights
        attention_weights = attn_weights
        context_vec = attn_weights @ values
        return context_vec



# torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(x))
# print(attention_weights)


context_len = attention_weights.shape[0]
mask_simple = torch.tril(torch.ones(context_len, context_len))
# print(mask_simple)

masked_attention_weights = attention_weights * mask_simple
# print(masked_attention_weights)

## Re normalise the masked rows to sum to 1
rows_sum = masked_attention_weights.sum(dim=-1, keepdim=True)
# print(rows_sum)
masked_attention_weights_normal = masked_attention_weights / rows_sum
# print(masked_attention_weights_normal)


## We are just using the softmax function to normalise the attention weights
## We can use the softmax function to do this
## We can convert the the values above the diagonal to -inf
## Then apply the softmax function
## This will give us the same result as above

mask = torch.triu(torch.ones(context_len, context_len), diagonal=1)
masked =attention_scores.masked_fill(mask.bool(), -torch.inf)
attention_weights = torch.softmax(masked / (d_out**0.5), dim=-1)
# print(attention_weights)

#Output


## Dropout
dropout = nn.Dropout(p=0.5)
example_tensor = torch.ones(6,6)
# print(dropout(example_tensor))


print(dropout(attention_weights))