import unittest
import numpy as np
import matplotlib.pyplot as plt
from src.interface_representation import InterfaceRepresentationType, sdf_to_diffuse_tanh, heaviside_to_sdf, convert_arr_from_heaviside


class TestInterfaceRepresentation(unittest.TestCase):

    def setUp(self):
        # Create a 3D volume with a linear interface represented by a Heaviside function
        self.volume_size = 256
        self.heaviside_volume = np.zeros((self.volume_size, self.volume_size, self.volume_size))
        self.heaviside_volume[:, :, :self.volume_size // 2] = 1

    def test_heaviside_to_sdf(self):
        sdf = heaviside_to_sdf(self.heaviside_volume)
        self.assertEqual(sdf.shape, self.heaviside_volume.shape)

        # Check that SDF is constant in y and z-direction
        self.assertTrue(np.all(np.diff(sdf[128, 128, :]) > 0.0001))
        self.assertTrue(np.all(np.diff(sdf[128, :, 128]) == 0))
        self.assertTrue(np.all(np.diff(sdf[:, 128, 128]) == 0))

    def test_sdf_to_diffuse_tanh(self):
        sdf = heaviside_to_sdf(self.heaviside_volume)
        diffuse_tanh = sdf_to_diffuse_tanh(sdf)
        self.assertEqual(diffuse_tanh.shape, self.heaviside_volume.shape)
        self.assertTrue(np.all(diffuse_tanh <= 1) and np.all(diffuse_tanh >= 0))

    def test_convert_arr_from_heaviside_to_diffuse_tanh(self):
        converted = convert_arr_from_heaviside(self.heaviside_volume, InterfaceRepresentationType.DIFFUSE_TANH)
        self.assertEqual(converted.shape, self.heaviside_volume.shape)
        self.assertTrue(np.all(converted <= 1) and np.all(converted >= 0))

    def test_convert_arr_from_heaviside_to_sdf(self):
        converted = convert_arr_from_heaviside(self.heaviside_volume, InterfaceRepresentationType.SIGNED_DISTANCE)
        self.assertEqual(converted.shape, self.heaviside_volume.shape)
        self.assertTrue(np.all(converted <= 1) and np.all(converted >= -1))

if __name__ == '__main__':
    unittest.main()