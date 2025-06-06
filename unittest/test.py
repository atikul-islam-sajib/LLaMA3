import os
import sys
import torch
import warnings
import unittest
import torch.nn as nn

sys.path.append("./src/")

try:
    from rms_norm import RMSNorm
    from activation_func import SwiGLU
    from positional_encoding import RoPE
    from attention import GroupedQueryAttention
    from feedforward import FeedForwardNeuralNetwork
    from transformer_block import TransformerBlock
    from model import LLaMA3
except ImportError:
    warnings.warn("Unable to import modules".capitalize())
    sys.exit(1)


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
        self.network = FeedForwardNeuralNetwork(
            hidden_dimension=self.dimension_size,
            output_dimension=self.dimension_size * 4,
        )
        self.positional_encoding = RoPE(
            dimension=self.dimension_size, sequence_length=self.sequence_length
        )
        self.transformer = TransformerBlock(
            dimension=self.dimension_size,
            query_heads=self.query_heads,
            kv_heads=self.kv_heads,
            sequence_length=self.sequence_length,
            output_dimension=self.dimension_size * 4,
        )
        self.model = LLaMA3(
            dimension=self.dimension_size,
            num_vocabularies=4096,
            query_heads=self.query_heads,
            num_layers=1,
            kv_heads=self.kv_heads,
            eps=1e-4,
            sequence_length=self.sequence_length,
            base=10000,
            output_dimension=self.dimension_size * 4,
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

    def test_mlp_layer(self):
        texts = torch.randn(
            (self.batch_size, self.sequence_length, self.dimension_size)
        )
        self.assertEqual(
            self.network(texts).shape,
            texts.shape,
            "FeedForwardNeuralNetwork is not working properly".capitalize(),
        )

        self.assertIsInstance(
            self.network(texts),
            torch.Tensor,
            "FeedForwardNeuralNetwork is not working properly".capitalize(),
        )

    def test_positional_encoding(self):
        texts = torch.randn(
            (self.batch_size, self.sequence_length, self.dimension_size)
        )
        self.assertEqual(
            self.positional_encoding(texts).shape,
            texts.shape,
            "RoPE is not working properly".capitalize(),
        )

        self.assertIsInstance(
            self.positional_encoding(texts),
            torch.Tensor,
            "RoPE is not working properly".capitalize(),
        )
        
    def test_transformer_block(self):
        texts = torch.randn(
            (self.batch_size, self.sequence_length, self.dimension_size)
        )
        self.assertEqual(
            self.transformer(texts).shape,
            texts.shape,
            "TransformerBlock is not working properly".capitalize(),
        )

        self.assertIsInstance(
            self.transformer(texts),
            torch.Tensor,
            "TransformerBlock is not working properly".capitalize(),
        )
        
    def test_model(self):
        texts = torch.randn(
            (self.batch_size, self.sequence_length, self.dimension_size)
        )
        output = torch.randn((self.batch_size, 4096))
        self.assertEqual(
            self.model(texts).shape,
            output.shape,
            "LLaMA3 is not working properly".capitalize(),
        )

        self.assertIsInstance(
            self.model(texts),
            torch.Tensor,
            "LLaMA3 is not working properly".capitalize(),
        )


if _name_ == "_main_":
    unittest.main()