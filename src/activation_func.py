import os
import sys
import torch
import torch.nn as nn

sys.path.append("./src/")

class SwiGLU(nn.Module):
    def __init__(self, name: str = "SwiGLU"):
        super(SwiGLU, self).__init__()
        
    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        pass
    
if __name__ == "__main__":
    pass