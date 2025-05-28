import os
import sys
import torch
import torch.nn as nn

sys.path.append("./src/")

class RMSNormalization(nn.Module):
    def __init__(self, dimension: int = 512, eps: float = 1e-4):
        super(RMSNormalization, self).__init__()
        
        self.dimension = dimension
        self.eps = eps
        
    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor".capitalize())
        
        pass
    
    
if __name__ == "__main__":
    pass