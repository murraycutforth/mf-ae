import unittest
from pathlib import Path

import numpy as np
import torch
from src.datasets.ellipse_dataset import EllipseDataset
from src.interface_representation.utils import InterfaceRepresentationType
from src.plotting_utils import write_slice_plot, write_isosurface_plot_from_arr
from src.datasets.ellipse_dataset import generate_ellipsoid_sdf


class TestGenerateEllipsoidSDF(unittest.TestCase):

    def test_sdf_shape(self):
        center = np.array([0.5, 0.5, 0.5])
        radii = np.array([0.3, 0.3, 0.5])
        N = 64
        sdf = generate_ellipsoid_sdf(center, radii, N)
        self.assertEqual(sdf.shape, (N, N, N))

    def test_sdf_center_value(self):
        center = np.array([0.5, 0.5, 0.5])
        radii = np.array([0.3, 0.3, 0.5])
        N = 64
        sdf = generate_ellipsoid_sdf(center, radii, N)
        center_idx = (N // 2 - 1, N // 2 - 1, N // 2 - 1)
        self.assertAlmostEqual(sdf[center_idx], -0.3, places=1)

    def test_sdf_min_max_values(self):
        center = np.array([0.5, 0.5, 0.5])
        radii = np.array([0.3, 0.3, 0.5])
        N = 64
        sdf = generate_ellipsoid_sdf(center, radii, N)
        self.assertLessEqual(np.min(sdf), -0.25)
        self.assertGreaterEqual(np.max(sdf), 0.5)

    def test_sdf_boundary_values(self):
        center = np.array([0.5, 0.5, 0.5])
        radii = np.array([0.3, 0.3, 0.5])
        N = 64
        sdf = generate_ellipsoid_sdf(center, radii, N)
        boundary_idx = (0, 0, 0)
        self.assertGreater(sdf[boundary_idx], 0.0)

    def test_gradient_magnitude(self):
        center = np.array([0.5, 0.5, 0.5])
        radii = np.array([0.3, 0.3, 0.5])
        N = 64
        dx = 1.0 / N
        sdf = generate_ellipsoid_sdf(center, radii, N)

        grad = np.gradient(sdf)

        grad_mag = np.sqrt((grad[0] / dx)**2 + (grad[1] / dx)**2 + (grad[2] / dx)**2)

        # Plot slices through the gradient magnitude
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(sdf[N // 2, :, :])
        plt.colorbar()
        plt.title("SDF Slice")
        plt.show()

        plt.figure()
        plt.imshow(grad_mag[N // 2, :, :])
        plt.colorbar()
        plt.title("Gradient Magnitude Slice")
        plt.show()

        mean_grad_mag = np.mean(grad_mag)
        median_grad_mag = np.median(grad_mag)
        self.assertAlmostEqual(mean_grad_mag, 1.0, places=1)
        self.assertAlmostEqual(median_grad_mag, 1.0, places=2)
        self.assertLess(np.max(grad_mag), 1.25)


class TestEllipseDataset(unittest.TestCase):

    def setUp(self):
        self.num_samples = 10
        self.vol_size = 64
        self.interface_reps = [
            InterfaceRepresentationType.TANH,
            InterfaceRepresentationType.SDF
        ]
        self.epsilon = 0.02

    def test_initialization(self):
        for interface_rep in self.interface_reps:
            with self.subTest(interface_rep=interface_rep):
                dataset = EllipseDataset(
                    debug=False,
                    num_samples=self.num_samples,
                    vol_size=self.vol_size,
                    interface_rep=interface_rep,
                    epsilon=self.epsilon
                )
                self.assertEqual(len(dataset), self.num_samples)
                self.assertEqual(dataset.vol_size, self.vol_size)
                self.assertEqual(dataset.interface_rep, interface_rep)

    def test_data_generation(self):
        for interface_rep in self.interface_reps:
            with self.subTest(interface_rep=interface_rep):
                dataset = EllipseDataset(
                    debug=False,
                    num_samples=self.num_samples,
                    vol_size=self.vol_size,
                    interface_rep=interface_rep,
                    epsilon=self.epsilon
                )
                for data in dataset:
                    self.assertEqual(data.shape, (1, self.vol_size, self.vol_size, self.vol_size))
                    self.assertTrue(torch.is_tensor(data))

    def test_normalization(self):
        for interface_rep in self.interface_reps:
            with self.subTest(interface_rep=interface_rep):
                dataset = EllipseDataset(
                    debug=False,
                    num_samples=self.num_samples,
                    vol_size=self.vol_size,
                    interface_rep=interface_rep,
                    epsilon=self.epsilon
                )
                min_val = np.min([torch.min(d).item() for d in dataset.data])
                max_val = np.max([torch.max(d).item() for d in dataset.data])
                self.assertAlmostEqual(min_val, 0.0, places=5)
                self.assertAlmostEqual(max_val, 1.0, places=5)

    def test_getitem(self):
        for interface_rep in self.interface_reps:
            with self.subTest(interface_rep=interface_rep):
                dataset = EllipseDataset(
                    debug=False,
                    num_samples=self.num_samples,
                    vol_size=self.vol_size,
                    interface_rep=interface_rep,
                    epsilon=self.epsilon
                )
                for i in range(len(dataset)):
                    data = dataset[i]
                    self.assertEqual(data.shape, (1, self.vol_size, self.vol_size, self.vol_size))
                    self.assertTrue(torch.is_tensor(data))

    def test_print_data_examples(self):
        for interface_rep in self.interface_reps:
            with self.subTest(interface_rep=interface_rep):

                outdir = Path('test_ellipse_dataset_output') / interface_rep.name
                outdir.mkdir(parents=True, exist_ok=True)

                dataset = EllipseDataset(
                    debug=False,
                    num_samples=self.num_samples,
                    vol_size=self.vol_size,
                    interface_rep=interface_rep,
                    epsilon=0.05,
                )
                for i in range(5):
                    data = dataset[i]
                    data = data.squeeze().numpy()

                    write_slice_plot(outdir / f'data_{i}.png', data)

                    data_heaviside = dataset.unnormalise_array(data)

                    write_slice_plot(outdir / f'data_{i}_unnormalised.png', data_heaviside)
                    #write_isosurface_plot_from_arr(data_heaviside, dx=dataset.vol_size, outname=outdir / f'data_{i}_isosurface.png', level=0.5, verbose=False)


if __name__ == '__main__':
    unittest.main()