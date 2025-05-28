import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")


class RMSNorm(nn.Module):
    def __init__(self, dimension: int = 512, eps: float = 1e-4):
        super(RMSNorm, self).__init__()

        self.dimension = dimension
        self.eps = eps

        self.gamma = nn.Parameter(
            data=torch.ones((1, 1, self.dimension)), requires_grad=True
        )

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor".capitalize())

        RMS = torch.sqrt(input=torch.mean(input=x**2, dim=-1) + self.eps)
        RMS = RMS.unsqueeze(dim=-1)

        RMSNorm = x / RMS

        return torch.mul(input=RMSNorm, other=self.gamma)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RMSNorm activation function".title())
    parser.add_argument(
        "--dimension",
        type=int,
        default=512,
        help="Dimension of the input tensor".capitalize(),
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-4,
        help="Epsilon value for numerical stability".capitalize(),
    )
    args = parser.parse_args()

    dimension = args.dimension
    eps = args.eps

    norm = RMSNorm(dimension=dimension, eps=eps)

    batch_size = 64
    sequence_length = 128
    dimension_size = 512

    assert (norm(torch.randn(batch_size, sequence_length, dimension)).size()) == (
        batch_size,
        sequence_length,
        dimension,
    ), "RMSNorm activation function is not working properly".capitalize()
