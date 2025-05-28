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

        self.gamma = nn.Parameter(data=torch.ones((self.dimension // self.dimension,
                                  self.dimension // self.dimension, self.dimension)), requires_grad=True)

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor".capitalize())

        RMS = torch.sqrt(torch.mean(x ** 2, dim=-1) + self.eps)
        RMS = RMS.unsqueeze(dim=-1)
        
        RMSNorm = x / RMS
        
        return torch.mul(RMSNorm, self.gamma)
        


if __name__ == "__main__":
    norm = RMSNormalization(dimension=512)
    
    print(norm(torch.randn(64, 128, 512)).size())
