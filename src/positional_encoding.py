import os
import sys
import torch
import torch.nn as nn

sys.path.append("./src/")

class RoPE(nn.Module):
    def __init__(self, dimension: int = 512, sequence_length: int = 128):
        super(RoPE, self).__init__()
        
        self.dimension = dimension
        self.sequence_length = sequence_length
        
    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor".capitalize())
        
if __name__ == "__main__":
    pass