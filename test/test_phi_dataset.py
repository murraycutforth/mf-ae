import unittest
import numpy as np
from src.datasets.phi_field_dataset import PhiDataset
from src.interface_representation.utils import InterfaceRepresentationType
from src.paths import project_dir

class TestPhiDataset(unittest.TestCase):

    def setUp(self):
        self.data_dir = project_dir() / 'data'
        self.split = 'train'
        self.epsilon = 1/256

    def test_dataset_initialization_tanh(self):
        interface_rep = InterfaceRepresentationType.TANH
        dataset = PhiDataset(data_dir=self.data_dir, split=self.split, interface_rep=interface_rep, epsilon=self.epsilon)
        self.assertEqual(len(dataset), len(dataset.filenames))
        self.assertEqual(dataset.data[0].shape, (1, 256, 256, 256))

    def test_plot_first_volume_sdf(self):
        interface_rep = InterfaceRepresentationType.SDF_APPROX
        dataset = PhiDataset(data_dir=self.data_dir, split=self.split, interface_rep=interface_rep)
        volume = dataset.data[0][0].numpy()
        import matplotlib.pyplot as plt
        plt.imshow(volume[128, 50:100, 100:125])
        plt.colorbar()
        plt.show()

    def test_plot_first_volume_tanh(self):
        interface_rep = InterfaceRepresentationType.TANH
        dataset = PhiDataset(data_dir=self.data_dir, split=self.split, interface_rep=interface_rep, epsilon=self.epsilon)
        volume = dataset.data[0][0].numpy()
        import matplotlib.pyplot as plt
        plt.imshow(volume[128, 50:100, 100:125])
        plt.colorbar()
        plt.show()

if __name__ == '__main__':
    unittest.main()