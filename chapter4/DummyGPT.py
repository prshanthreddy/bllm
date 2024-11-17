import torch
import torch.nn as nn
from dict import GPT_CONFIG_124M as cfg

class DummyGPT(nn.Module):
    def __init__(self,cfg):
        super.__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"],cfg["emb_dim"])
        self.pos_emb = nn.Parameter(torch.zeros(1,cfg["context_size"],cfg["emb_dim"]))
        self.drop_emb = nn.Dropout(cfg["dropout"])
        self.trf_blocks= nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layer"])]
        )
        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head=nn.Linear(cfg["emb_dim"],cfg["vocab_size"],bias=False)

    def forward(self,in_idx):
        batch_size,seq_len = in_idx.size()
        tok_emb = self.tok_emb(in_idx)
        pos_emb = self.pos_emb[
            torch.arange(seq_len,device=in_idx.device)
        ]
        x = self.drop_emb(tok_emb+pos_emb)
        x = self.trf_blocks(x)
        x=slf.final_norm(x)
        x=self.out_head(x)
        return x

class DummyTransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
    def forward(self,x):
        return x
    
class DummyLayerNorm(nn.Module):
    def __init__(self,emb_dim,eps=1e-5):
        super().__init__()
    def forward(self,x):
        return x

if __name__ == "__main__":
    x = torch.randint(0,50257,(2,3))
    model = DummyGPT(cfg)
    print(model(x))
    print(model(x).shape)
    print(model(x).size())
    print(model(x).size(0))
    print(model(x).size(1))
