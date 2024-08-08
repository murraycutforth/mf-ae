import unittest
import torch
import torch.nn as nn
from src.models.conv_ae import ConvBlockDown, ConvBlockUp, ConvAutoencoderBaseline


class TestConvBlockDown(unittest.TestCase):
    def setUp(self):
        self.block = ConvBlockDown(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1, activation=nn.ReLU(), norm=nn.BatchNorm3d)
        self.input_tensor = torch.randn(1, 1, 64, 64, 64)

    def test_construction(self):
        self.assertIsInstance(self.block, ConvBlockDown)

    def test_forward_pass(self):
        output = self.block(self.input_tensor)
        self.assertEqual(output.shape, (1, 32, 32, 32, 32))


class TestConvBlockUp(unittest.TestCase):
    def setUp(self):
        self.block = ConvBlockUp(in_channels=32, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1, activation=nn.ReLU(), norm=nn.BatchNorm3d)
        self.input_tensor = torch.randn(1, 32, 32, 32, 32)

    def test_construction(self):
        self.assertIsInstance(self.block, ConvBlockUp)

    def test_forward_pass(self):
        output = self.block(self.input_tensor)
        self.assertEqual(output.shape, (1, 1, 64, 64, 64))


class TestConvAutoencoderBaselineFlat(unittest.TestCase):
    def setUp(self):
        self.model = ConvAutoencoderBaseline(image_shape=(64, 64, 64), flat_bottleneck=True, latent_dim=100, activation=nn.ReLU(), norm=nn.BatchNorm3d)
        self.input_tensor = torch.randn(1, 1, 64, 64, 64)

    def test_construction(self):
        self.assertIsInstance(self.model, ConvAutoencoderBaseline)

    def test_forward_pass(self):
        output = self.model(self.input_tensor)
        self.assertEqual(output.shape, self.input_tensor.shape)


class TestConvAutoencoderBaseline(unittest.TestCase):
    def setUp(self):
        self.model = ConvAutoencoderBaseline(image_shape=(64, 64, 64), flat_bottleneck=False, latent_dim=100, activation=nn.ReLU(), norm=nn.BatchNorm3d)
        self.input_tensor = torch.randn(1, 1, 64, 64, 64)

    def test_construction(self):
        self.assertIsInstance(self.model, ConvAutoencoderBaseline)

    def test_forward_pass(self):
        output = self.model(self.input_tensor)
        self.assertEqual(output.shape, self.input_tensor.shape)


if __name__ == '__main__':
    unittest.main()