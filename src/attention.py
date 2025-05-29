import os
import sys
import torch
import torch.nn as nn

sys.path.append("./src/")


class GroupedQueryAttention(nn.Module):
    def __init__(self, dimension: int = 512, query_heads: int = 8, kv_heads: int = 4):
        super(GroupedQueryAttention, self).__init__()

        self.dimension = dimension
        self.query_heads = query_heads
        self.kv_heads = kv_heads

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

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor".capitalize())

        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

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


if __name__ == "__main__":
    attention = GroupedQueryAttention()
    texts = torch.randn((64, 128, 512))

    query = attention(texts)
    print(query.size())
