import os
import sys
import torch
import torch.nn as nn

sys.path.append("./src/")

from rms_norm import RMSNorm
from attention import GroupedQueryAttention
from feedforward import FeedForwardNeuralNetwork


class TransformerBlock(nn.Module):
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
        super(TransformerBlock, self).__init__()

        self.dimension = dimension
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.eps = eps
        self.sequence_length = sequence_length
        self.base = base
        self.output_dimension = output_dimension

        self.attention_norm = RMSNorm(dimension=self.dimension, eps=self.eps)
        self.feedforward_norm = RMSNorm(dimension=self.dimension, eps=self.eps)
        
        self.attention = GroupedQueryAttention(
            dimension=self.dimension,
            query_heads=self.query_heads,
            kv_heads=self.kv_heads,
            sequence_length=self.sequence_length
        )
        self.feedforward_network = FeedForwardNeuralNetwork(
            hidden_dimension=self.dimension, output_dimension=self.output_dimension
        )

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor".capitalize())
        
        residual = x
        
        x1 = self.attention_norm(x)
        x1 = self.attention(x1)
        x1 = torch.add(input=residual, other=x1)
        
        residual = x1
        
        x2 = self.feedforward_norm(x1)
        x2 = self.feedforward_network(x2)
        x2 = torch.add(input=residual, other=x2)
        
        return x2


if __name__ == "__main__":
    transformer = TransformerBlock(
        dimension=512,
        query_heads=8,
        kv_heads=4,
        sequence_length=128,
        base=10000,
        output_dimension=14336,
    )

    input_tensor = torch.randn(64, 128, 512)
    output_tensor = transformer(input_tensor)

    print(
        "TransformerBlock is working properly"
        if output_tensor.size() == (64, 128, 512)
        else "TransformerBlock is not working properly".capitalize()
    )
