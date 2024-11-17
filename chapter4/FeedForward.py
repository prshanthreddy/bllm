import torch
import torch.nn as nn
import dict
from GELU import GELU
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

if __name__ == "__main__":
    x = torch.randn(2,3, 768)
    feed_forward = FeedForward(dict.GPT_CONFIG_124M)
    print(feed_forward(x))
    print(feed_forward(x).shape)
    print(feed_forward(x).size())
    print(f