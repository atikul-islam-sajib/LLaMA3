import os
import sys
import math
import torch
import torch.nn as nn

sys.path.append("./src/")

class RoPE(nn.Module):
    def __init__(self, dimension: int = 512, sequence_length: int = 128, base: int = 10000):
        super(RoPE, self).__init__()
        
        self.dimension = dimension//2
        self.sequence_length = sequence_length
        self.base = base
        
        self.sin_values = torch.zeros((self.sequence_length, self.dimension))
        self.cos_values = torch.zeros((self.sequence_length, self.dimension))
        
        for position in range(self.sequence_length):
            for i in range(self.dimension):
                inverse_frequncy = 1.0 / (self.base ** (2 * (i // 2) / self.dimension))
                
                theta = position * inverse_frequncy
                
                self.sin_values[position, i] = math.sin(theta)
                self.cos_values[position, i] = math.cos(theta)
                
        self.register_buffer("sin", self.sin_values.unsqueeze(dim=0))
        self.register_buffer("cos", self.cos_values.unsqueeze(dim=0))
        
    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor".capitalize())
        
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        
        sin = self.sin[:, :x.size(1), :]
        cos = self.cos[:, :x.size(1), :]

        rotated_even = x1 * cos - x2 * sin
        rotated_odd  = x1 * sin + x2 * cos

        
        output = torch.stack((rotated_even, rotated_odd), dim=-1)
        output = output.view(output.size(0), output.size(1), -1)
        
        return output
        
        
if __name__ == "__main__":
    encoding = RoPE(dimension=512, sequence_length=128)
    
    texts = torch.randn((64, 128, 512))
    
    print(encoding(texts).size())