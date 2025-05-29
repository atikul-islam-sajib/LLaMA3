import os
import sys
import torch
import torch.nn as nn

sys.path.append("./src/")

class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, hidden_dimension: int = 4096, output_dimension: int = 14336, bias: bool = True):
        super(FeedForwardNeuralNetwork, self).__init__()
        
        self.hidden_dimension = hidden_dimension
        self.output_dimension = output_dimension
        self.bias = bias
        
    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        
if __name__ == "__main__":
    pass