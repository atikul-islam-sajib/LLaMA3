import os
import sys
import tqdm
import torch
import warnings
import torch.nn as nn

sys.path.append("./src/")

try:
    from rms_norm import RMSNorm
    from transformer_block import TransformerBlock
except ImportError:
    warnings.warn("Unable to import modules".capitalize())
    sys.exit(1)


class LLaMA3(nn.Module):
    def __init__(
        self,
        dimension: int = 512,
        num_vocabularies: int = 4096,
        query_heads: int = 8,
        num_layers: int = 16,
        kv_heads: int = 4,
        eps: float = 1e-4,
        sequence_length: int = 128,
        base: int = 10000,
        output_dimension: int = 14336,
    ):
        super(LLaMA3, self).__init__()

        self.dimension = dimension
        self.num_vocabularies = num_vocabularies
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.num_layers = num_layers
        self.eps = eps
        self.sequence_length = sequence_length
        self.base = base
        self.output_dimension = output_dimension

        self.ouput_layers = []

        self.transformer_layers = nn.Sequential(
            *[
                TransformerBlock(
                    dimension=dimension,
                    query_heads=query_heads,
                    kv_heads=kv_heads,
                    sequence_length=sequence_length,
                    base=base,
                    output_dimension=output_dimension,
                )
                for _ in range(self.num_layers)
            ]
        )

        for index in range(2):
            self.ouput_layers += [
                nn.Linear(in_features=self.dimension, out_features=self.dimension)
            ]
            if index != 0:
                self.ouput_layers += [
                    nn.Linear(
                        in_features=self.dimension, out_features=self.num_vocabularies
                    )
                ]
                self.ouput_layers += [nn.Softmax(dim=-1)]

        self.output = nn.Sequential(*self.ouput_layers)
        
        self.rms_norm = RMSNorm(dimension=self.dimension, eps=self.eps)

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor".capitalize())

        for layer in self.transformer_layers:
            x = layer(x)

        x = self.rms_norm(x)
        x = torch.mean(input=x, dim=1)

        output = self.output(x)

        return output


if __name__ == "__main__":
    model = LLaMA3(
        dimension=512,
        num_vocabularies=4096,
        query_heads=8,
        num_layers=2,
        kv_heads=4,
        eps=1e-4,
        sequence_length=128,
        base=10000,
        output_dimension=14336,
    )

    x = torch.randn((1, 128, 512))

    print(model(x).shape)
