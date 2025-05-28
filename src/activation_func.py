import os
import sys
import math
import torch
import torch.nn as nn

sys.path.append("./src/")


class SwiGLU(nn.Module):
    def __init__(self, name: str = "SwiGLU"):
        super(SwiGLU, self).__init__()

        self.name = name
        self.constant = 0.044715

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")

        swish = x * torch.sigmoid(x)
        gelu = 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / math.pi)) * (x + self.constant * torch.pow(x, 3))))
        return swish * gelu


if __name__ == "__main__":
    activation_func = SwiGLU()

    texts = torch.randn((64, 128, 512))
    print(activation_func(texts).size())
