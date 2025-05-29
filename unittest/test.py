import os
import sys
import torch
import unittest
import torch.nn as nn

sys.path.append("./src/")

from rms_norm import RMSNorm
from activation_func import SwiGLU
from attention import GroupedQueryAttention


class UnitTest(unittest.TestCase):
    def setUp(self):
        self.batch_size = 64
        self.sequence_length = 128
        self.dimension_size = 512
        self.query_heads = 8
        self.kv_heads = 4

        self.activation_func = SwiGLU()
        self.rms_normalization = RMSNorm(dimension=self.dimension_size)
        self.attention = GroupedQueryAttention(
            dimension=self.dimension_size,
            query_heads=self.query_heads,
            kv_heads=self.kv_heads,
        )

    def test_activation_func(self):
        texts = torch.randn(
            (self.batch_size, self.sequence_length, self.dimension_size)
        )

        self.assertEqual(
            self.activation_func(texts).shape,
            texts.shape,
            "SwiGLU activation function is not working properly".capitalize(),
        )

        self.assertIsInstance(
            self.activation_func(texts),
            torch.Tensor,
            "SwiGLU activation function is not working properly".capitalize(),
        )

    def test_rms_norm(self):
        texts = torch.randn(
            (self.batch_size, self.sequence_length, self.dimension_size)
        )

        self.assertEqual(
            self.rms_normalization(texts).shape,
            texts.shape,
            "RMSNorm activation function is not working properly".capitalize(),
        )

        self.assertIsInstance(
            self.rms_normalization(texts),
            torch.Tensor,
            "RMSNorm activation function is not working properly".capitalize(),
        )

    def test_attention_layer(self):
        texts = torch.randn(
            (self.batch_size, self.sequence_length, self.dimension_size)
        )

        self.assertEqual(
            self.attention(texts).shape,
            texts.shape,
            "Attention layer is not working properly".capitalize(),
        )

        self.assertIsInstance(
            self.attention(texts),
            torch.Tensor,
            "Attention layer is not working properly".capitalize(),
        )


if __name__ == "__main__":
    unittest.main()
