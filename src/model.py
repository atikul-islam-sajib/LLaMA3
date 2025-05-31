import os
import sys
import torch
import torch.nn as nn

sys.path.append("./src/")


class LLaMA3(nn.Module):
    def __init__(
        self,
        dimension: int = 512,
        query_heads: int = 8,
        kv_heads: int = 4,
        eps: float = 1e-4,
        sequence_length: int = 128,
        base: int = 10000,
        output_dimension: int = 14336,
    ):
        super(LLaMA3, self).__init__()
        
        self.dimension = dimension
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.eps = eps
        self.sequence_length = sequence_length
        self.base = base
        self.output_dimension = output_dimension
        
    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor".capitalize())
        
        
if __name__ == "__main__":
    pass
