import os
import sys
import torch
import unittest
import torch.nn as nn

sys.path.append("./src/")

from activation_func import SwiGLU
from RMSNorm import RMSNormalization


class UnitTest(unittest.TestCase):
    def setUp(self):
        self.batch_size = 64
        self.sequence_length = 128
        self.dimension_size = 512

        self.activation_func = SwiGLU()
        self.rms_normalization = RMSNormalization(dimension=self.dimension_size)

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


if __name__ == "__main__":
    unittest.main()
