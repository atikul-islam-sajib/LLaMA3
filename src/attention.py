import os
import sys
import torch
import argparse
import warnings
import torch.nn as nn
from torchview import draw_graph

sys.path.append("./src/")

from positional_encoding import RoPE


class GroupedQueryAttention(nn.Module):
    def __init__(self, dimension: int = 512, query_heads: int = 8, kv_heads: int = 4, sequence_length: int = 128):
        super(GroupedQueryAttention, self).__init__()

        self.dimension = dimension
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.sequence_length = sequence_length

        warnings.warn(
            """If you are defined the query heads = 8 and kv heads = 4, it is recommended to use the SwiGLU activation function""",
            UserWarning,
        )

        assert (
            self.dimension % self.query_heads == 0
        ), "Dimension must be divisible by query heads".capitalize()
        assert (
            self.dimension % self.kv_heads == 0
        ), "Dimension must be divisible by kv heads".capitalize()

        self.head_dim = self.dimension // self.query_heads
        self.num_of_repeatation = self.query_heads // self.kv_heads

        self.query = nn.Linear(
            in_features=self.dimension,
            out_features=self.query_heads * self.head_dim,
            bias=False,
        )
        self.key = nn.Linear(
            in_features=self.dimension,
            out_features=self.kv_heads * self.head_dim,
            bias=False,
        )
        self.value = nn.Linear(
            in_features=self.dimension,
            out_features=self.kv_heads * self.head_dim,
            bias=False,
        )
        self.output = nn.Linear(
            in_features=self.dimension, out_features=self.dimension, bias=False
        )
        self.Q_positional_encoding = RoPE(
            dimension=self.head_dim * self.query_heads,
            sequence_length=self.dimension * 2,
        )
        self.K_positional_encoding = RoPE(
            dimension=self.head_dim * self.kv_heads, sequence_length=self.sequence_length
        )

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor".capitalize())

        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        query = self.Q_positional_encoding(query)
        key = self.K_positional_encoding(key)

        assert (
            key.size() == value.size()
        ), "Key and value must have the same size".capitalize()

        query = query.view(
            query.size(0),
            query.size(1),
            self.query_heads,
            query.size(-1) // self.query_heads,
        )
        key = key.view(
            key.size(0), key.size(1), self.kv_heads, key.size(-1) // self.kv_heads
        )
        value = value.view(
            value.size(0), value.size(1), self.kv_heads, value.size(-1) // self.kv_heads
        )

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        key = torch.repeat_interleave(input=key, repeats=self.num_of_repeatation, dim=1)
        value = torch.repeat_interleave(
            input=value, repeats=self.num_of_repeatation, dim=1
        )

        attention = torch.matmul(
            query, torch.transpose(input=key, dim0=-1, dim1=-2)
        ) / torch.sqrt(torch.tensor(self.head_dim))
        attention = torch.softmax(input=attention, dim=-1)

        attention = torch.matmul(input=attention, other=value)
        attention = torch.permute(input=attention, dims=(0, 2, 1, 3))

        attention = attention.reshape(
            attention.size(0), attention.size(1), attention.size(2) * attention.size(3)
        )

        attention = self.output(attention)

        return attention

    @staticmethod
    def total_parameters(model):
        if not isinstance(model, GroupedQueryAttention):
            raise TypeError("Model must be a GroupedQueryAttention".capitalize())

        return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grouped Query Attention".title())
    parser.add_argument(
        "--dimension", type=int, default=512, help="Dimension of the input".capitalize()
    )
    parser.add_argument(
        "--query_heads", type=int, default=8, help="Number of query heads".capitalize()
    )
    parser.add_argument(
        "--kv_heads", type=int, default=4, help="Number of kv heads".capitalize()
    )
    parser.add_argument(
        "--display", action="store_true", help="Display the graph".capitalize()
    )
    parser.add_argument(
        "--params", action="store_true", help="Display the parameters".capitalize()
    )

    args = parser.parse_args()

    dimension = args.dimension
    query_heads = args.query_heads
    kv_heads = args.kv_heads

    batch_size = 64
    sequence_length = 128
    dimension_size = 512

    attention = GroupedQueryAttention(
        dimension=dimension, query_heads=query_heads, kv_heads=kv_heads
    )

    texts = torch.randn((batch_size, sequence_length, dimension_size))
    output = attention(texts)

    assert output.size() == (
        batch_size,
        sequence_length,
        dimension_size,
    ), "GroupedQueryAttention is not working properly".capitalize()

    if args.display:
        draw_graph(model=attention, input_data=texts).visual_graph.render(
            filename="./artifacts/files/GQA", format="png"
        )
        print("Image saved in the folder ./artifacts/files/GQA.png".capitalize())

    if args.params:
        print(
            "Total paramaters of the GQA = {}".format(
                GroupedQueryAttention.total_parameters(model=attention)
            )
        )
