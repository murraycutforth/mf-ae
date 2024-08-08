import json
import math
from pathlib import Path
import logging

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.evaluation.eval_ae_error import evaluate_autoencoder


logger = logging.getLogger(__name__)


class MyAETrainer():
    """High level class for training a multiphase flow autoencoder. Provides interface for training, saving, and loading
    models. Also includes plotting functions.
    """
    def __init__(
            self,
            model: nn.Module,
            dataset: Dataset,
            dataset_val: Dataset,
            train_batch_size = 32,
            train_lr = 1e-4,
            train_num_epochs = 100,
            adam_betas = (0.9, 0.99),
            save_and_sample_every = 10,
            results_folder = './results',
            amp = False,
            mixed_precision_type = 'fp16',
            cpu_only = False,
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches=False,
            mixed_precision=mixed_precision_type if amp else 'no',
            cpu=cpu_only,
        )

        self.model = model
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = train_batch_size
        self.train_num_epochs = train_num_epochs
        self.epoch = 0

        # dataset and dataloader
        dl = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=0)
        dl_val = DataLoader(dataset_val, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)

        # Check the size of dataset images
        first_batch = next(iter(dl))
        assert len(first_batch.shape) == 5, 'Expected 4D tensor for 3D convolutional model'

        self.opt = Adam(model.parameters(), lr=train_lr, betas=adam_betas)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # Move model and data to device
        self.dl = self.accelerator.prepare(dl)
        self.dl_val = self.accelerator.prepare(dl_val)
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'epoch': self.epoch,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.diffusion_model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        loss_history = []

        with tqdm(initial = self.epoch, total = self.train_num_epochs, disable=not accelerator.is_main_process) as pbar:

            while self.epoch < self.train_num_epochs:

                for data in self.dl:
                    data = data.to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss_history.append(float(loss.item()))

                    self.accelerator.backward(loss)

                    pbar.set_description(f'loss: {loss.item():.4f}')

                    accelerator.wait_for_everyone()

                    self.opt.step()
                    self.opt.zero_grad()

                    accelerator.wait_for_everyone()

                self.epoch += 1

                if accelerator.is_main_process:

                    if self.epoch % self.save_and_sample_every == 0:

                        milestone = self.step // self.save_and_sample_every
                        self.save(milestone)

                        plot_samples(self.model, f'{milestone}', self.results_folder)

                        self.write_loss_history(loss_history)

                pbar.update(1)

        self.write_loss_history(loss_history)
        logger.info('training complete')

    def write_loss_history(self, loss_history):
        loss_history_path = self.results_folder / 'loss_history.json'
        with open(loss_history_path, 'w') as f:
            json.dump(loss_history, f)
        logger.info(f'Loss history written to {loss_history_path}')


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def exists(x):
    return x is not None


def plot_samples(model, dl_val, name: str, results_folder: str, n_samples: int = 96):

    all_samples = []
    all_data = []

    for _ in range(n_samples):
        data = next(dl_val)
        pred = model(data)

        all_samples.append(pred.cpu().numpy().squeeze())
        all_data.append(data.cpu().numpy().squeeze())

    all_samples = np.stack(all_samples, axis=0)
    all_data = np.stack(all_data, axis=0)

    np.savez_compressed(Path(results_folder) / f"{name}_samples.npz", all_samples=all_samples, all_data=all_data)
    logger.info(f'Saved samples to {results_folder}/{name}_samples.npz')

    logger.info(f'Sampled sequence shape: {all_samples[0].shape}')

    # Now compute error metrics on all val data and write out
    results = evaluate_autoencoder(model, dl_val)
    with open(Path(results_folder) / f"{name}_metrics.json", 'w') as f:
        json.dump(results, f)

    # Now just visualise slices from the first 9 samples

    fig, axs = plt.subplots(6, 3, figsize=(16, 12), dpi=200)
    for i, ax in enumerate(axs.flat):
        if i >= len(all_samples):
            break

        if i % 2 == 0:
            im = ax.imshow(all_samples[i // 2][:, :, 16], cmap="gray")
            fig.colorbar(im, ax=ax)
        else:
            im = ax.imshow(all_data[i // 2][:, :, 16], cmap="gray")
            fig.colorbar(im, ax=ax)

    fig.tight_layout()
    filename = Path(results_folder) / f"{name}_samples_spatial_slice.png"
    fig.savefig(filename)
    plt.close(fig)

    logger.info(f"Saved samples spatial slice plot to {filename}")

