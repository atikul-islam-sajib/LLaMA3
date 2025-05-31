import os
import sys
import torch
import argparse
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
            sequence_length=self.sequence_length,
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
    parser = argparse.ArgumentParser(description="Transformer Block for LLaMA".title())
    parser.add_argument(
        "--dimension",
        type=int,
        default=512,
        help="Dimension of the input tensor".capitalize(),
    )
    parser.add_argument(
        "--query_heads", type=int, default=8, help="Number of query heads".capitalize()
    )
    parser.add_argument(
        "--kv_heads", type=int, default=4, help="Number of kv heads".capitalize()
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=128,
        help="Sequence length of the input tensor".capitalize(),
    )
    parser.add_argument(
        "--base",
        type=int,
        default=10000,
        help="Base of the exponential function".capitalize(),
    )
    parser.add_argument(
        "--output_dimension",
        type=int,
        default=14336,
        help="Output dimension of the feedforward network".capitalize(),
    )

    args = parser.parse_args()

    dimension = args.dimension
    query_heads = args.query_heads
    kv_heads = args.kv_heads
    sequence_length = args.sequence_length
    base = args.base
    output_dimension = args.output_dimension

    batch_size = 64
    sequence_length = 128

    transformer = TransformerBlock(
        dimension=dimension,
        query_heads=query_heads,
        kv_heads=kv_heads,
        sequence_length=sequence_length,
        base=base,
        output_dimension=output_dimension,
    )

    input_tensor = torch.randn(batch_size, sequence_length, dimension)

    assert transformer(input_tensor).size() == (
        batch_size,
        sequence_length,
        dimension,
    ), "Transformer block is not working properly".capitalize()
