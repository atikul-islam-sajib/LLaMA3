import os
import sys
import torch
import torch.nn as nn

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


if __name__ == "__main__":
    network = FeedForwardNeuralNetwork(hidden_dimension=512, output_dimension=4 * 512)
    texts = torch.randn((64, 128, 512))
    print(network(texts).size())
