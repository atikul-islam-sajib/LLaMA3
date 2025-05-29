import os
import sys
import torch
import argparse
import torch.nn as nn
from torchview import draw_graph

sys.path.append("./src/")


class FeedForwardNeuralNetwork(nn.Module):
    def __init__(
        self,
        hidden_dimension: int = 4096,
        output_dimension: int = 14336,
        bias: bool = True,
    ):
        super(FeedForwardNeuralNetwork, self).__init__()

        self.hidden_dimension = hidden_dimension
        self.output_dimension = output_dimension
        self.bias = bias

        self.gate_projection = nn.Linear(
            in_features=self.hidden_dimension,
            out_features=self.output_dimension,
            bias=self.bias,
        )
        self.up_projection = nn.Linear(
            in_features=self.hidden_dimension,
            out_features=self.output_dimension,
            bias=self.bias,
        )
        self.down_projection = nn.Linear(
            in_features=self.output_dimension,
            out_features=self.hidden_dimension,
            bias=self.bias,
        )

        self.swish = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")

        gate_output = self.gate_projection(x)
        up_output = self.up_projection(x)
        up_output = self.swish(up_output)

        activation = torch.mul(input=gate_output, other=up_output)

        return self.down_projection(activation)

    @staticmethod
    def total_parameters(model):
        if not isinstance(model, FeedForwardNeuralNetwork):
            raise TypeError("Input must be a FeedForwardNeuralNetwork")

        return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MLP layer with SwiGLU activation function"
    )
    parser.add_argument(
        "--hidden_dimension",
        type=int,
        default=4096,
        help="Dimension of the hidden layer".capitalize(),
    )
    parser.add_argument(
        "--output_dimension",
        type=int,
        default=14336,
        help="Dimension of the output layer".capitalize(),
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the graph".capitalize(),
    )
    parser.add_argument(
        "--params", action="store_true", help="Display the parameters".capitalize()
    )

    args = parser.parse_args()

    hidden_dimension = args.hidden_dimension
    output_dimension = args.output_dimension

    batch_size = 64
    sequence_length = 128

    network = FeedForwardNeuralNetwork(
        hidden_dimension=hidden_dimension, output_dimension=output_dimension
    )

    texts = torch.randn((batch_size, sequence_length, hidden_dimension))

    assert (network(texts).size()) == (
        batch_size,
        sequence_length,
        hidden_dimension,
    ), "FeedForwardNeuralNetwork is not working properly".capitalize()

    if args.display:
        draw_graph(model=network, input_data=texts).visual_graph.render(
            filename="./artifacts/files/MLP", format="png"
        )
        print("Image saved in the folder ./artifacts/files/MLP.png".capitalize())

    if args.params:
        print(
            "Total paramaters of the MLP = {}".format(
                FeedForwardNeuralNetwork.total_parameters(model=network)
            )
        )
