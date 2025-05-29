import os
import sys
import torch
import torch.nn as nn

sys.path.append("./src/")

class GroupedQueryAttention(nn.Module):
    def __init__(self, dimension: int = 512, query_heads: int = 8, kv_heads: int = 4):
        super(GroupedQueryAttention, self).__init__()
        
        self.dimension = dimension
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        
        assert self.dimension % self.query_heads == 0, "Dimension must be divisible by query heads".capitalize()
        assert self.dimension % self.kv_heads == 0, "Dimension must be divisible by kv heads".capitalize()
        
    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor".capitalize())
        
if __name__ == "__main__":
    pass