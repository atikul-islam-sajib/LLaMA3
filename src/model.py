import os
import sys
import tqdm
import torch
import warnings
import argparse
import torch.nn as nn
from torchview import draw_graph

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

    @staticmethod
    def total_parameters(model):
        if not isinstance(model, LLaMA3):
            raise TypeError("Model must be a LLaMA3".capitalize())

        return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer Block for LLaMA".title())
    parser.add_argument(
        "--dimension",
        type=int,
        default=512,
        help="Dimension of the input tensor".capitalize(),
    )
    parser.add_argument(
        "--num_vocabularies",
        type=int,
        default=4096,
        help="Number of vocabularies".capitalize(),
    )
    parser.add_argument(
        "--num_layers", type=int, default=16, help="Number of layers".capitalize()
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
        "--eps", type=float, default=1e-4, help="Epsilon value".capitalize()
    )
    parser.add_argument(
        "--output_dimension",
        type=int,
        default=14336,
        help="Output dimension of the feedforward network".capitalize(),
    )
    parser.add_argument(
        "--display", action="store_true", help="Display the graph".capitalize()
    )
    parser.add_argument(
        "--params", action="store_true", help="Display the parameters".capitalize()
    )

    args = parser.parse_args()

    dimension = args.dimension
    num_vocabularies = args.num_vocabularies
    query_heads = args.query_heads
    kv_heads = args.kv_heads
    num_layers = args.num_layers
    sequence_length = args.sequence_length
    base = args.base
    eps = args.eps
    output_dimension = args.output_dimension
    display = args.display
    params = args.params

    model = LLaMA3(
        dimension=dimension,
        num_vocabularies=num_vocabularies,
        query_heads=query_heads,
        num_layers=num_layers,
        kv_heads=kv_heads,
        eps=eps,
        sequence_length=sequence_length,
        base=base,
        output_dimension=output_dimension,
    )

    x = torch.randn((sequence_length // sequence_length, sequence_length, dimension))

    assert (model(x).size()) == (
        sequence_length // sequence_length,
        num_vocabularies,
    ), "Output shape is not correct".capitalize()

    if display:
        draw_graph(model=model, input_data=x).visual_graph.render(
            filename="./artifacts/files/LLaMA3", format="png"
        )
        print("Image saved in the folder ./artifacts/files/LLaMA3.png".capitalize())

    if args.params:
        print(
            "Total paramaters of the LLaMA3 = {}".format(
                LLaMA3.total_parameters(model=model)
            )
        )