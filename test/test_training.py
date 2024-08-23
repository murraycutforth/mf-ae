import unittest
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.trainer.trainer import MyAETrainer
from src.models.conv_ae import ConvAutoencoderBaseline
from src.evaluation.eval_ae_error import evaluate_autoencoder


class TestTrainingProcessNoise(unittest.TestCase):
    def setUp(self):
        # Generate synthetic data
        self.train_ds = torch.randn(4, 1, 32, 32, 32)

        # Initialize model
        self.model = ConvAutoencoderBaseline(
            image_shape=(32, 32, 32),
            flat_bottleneck=False,
            #latent_dim=100,
            activation=nn.ReLU(),
            norm=nn.BatchNorm3d,
            feat_map_sizes=(16, 32, 64)
        )

        # Initialize trainer
        self.trainer = MyAETrainer(
            model=self.model,
            dataset=self.train_ds,
            dataset_val=self.train_ds,
            train_batch_size=1,
            train_lr=1e-3,
            train_num_epochs=1000,
            adam_betas=(0.9, 0.99),
            save_and_sample_every=500,
            results_folder='test_training_output',
            amp=False,
            mixed_precision_type='fp16',
            cpu_only=True,
            num_dl_workers=0,
        )

    def test_training(self):
        # Run training
        self.trainer.train()

        # Evaluate model
        results = evaluate_autoencoder(self.trainer.model, self.trainer.dl_val, 'test.csv', len(self.trainer.dl_val), return_metrics=True)

        self.assertLess(results['MAE'].mean(), 1.0)


class TestTrainingProcessSquares(unittest.TestCase):
    def setUp(self):
        # Generate synthetic data - samples with randomly placed squares
        self.train_ds = torch.zeros(10, 1, 32, 32, 32)
        for i in range(10):
            x = torch.randint(5, 20, (1,))
            y = torch.randint(5, 20, (1,))
            z = torch.randint(5, 20, (1,))
            self.train_ds[i, 0, x:x+5, y:y+5, z:z+5] = 1

        # Initialize model
        self.model = ConvAutoencoderBaseline(
            image_shape=(32, 32, 32),
            flat_bottleneck=False,
            #latent_dim=100,
            activation=nn.ReLU(),
            norm=nn.BatchNorm3d,
            feat_map_sizes=(16, 32, 64)
        )

        # Initialize trainer
        self.trainer = MyAETrainer(
            model=self.model,
            dataset=self.train_ds,
            dataset_val=self.train_ds,
            train_batch_size=10,
            train_lr=1e-3,
            train_num_epochs=500,
            adam_betas=(0.9, 0.99),
            save_and_sample_every=500,
            results_folder='test_training_output',
            amp=False,
            mixed_precision_type='fp16',
            cpu_only=True,
            num_dl_workers=0,
        )

    def test_training(self):
        # Run training
        self.trainer.train()

        # Evaluate model
        results = evaluate_autoencoder(self.trainer.model, self.trainer.dl_val, 'test.csv', len(self.trainer.dl_val), return_metrics=True)

        self.assertLess(results['MAE'].mean(), 1e-2)
        self.assertLess(results['MSE'].mean(), 1e-2)
        self.assertGreater(results['SSIM'].mean(), 0.8)



if __name__ == '__main__':
    unittest.main()