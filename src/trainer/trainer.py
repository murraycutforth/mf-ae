import json
import math
from pathlib import Path
import logging

import accelerate
import numpy as np
import pandas as pd
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
from src.plotting_utils import write_isosurface_plot_from_arr

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
            num_dl_workers = 0,
            loss: str = 'mse',
    ):
        super().__init__()

        self.accelerator = Accelerator(
            dataloader_config=accelerate.DataLoaderConfiguration(split_batches=False),
            mixed_precision=mixed_precision_type if amp else 'no',
            cpu=cpu_only,
        )

        self.model = model
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = train_batch_size
        self.train_num_epochs = train_num_epochs
        self.epoch = 0
        self.dataset_val = dataset_val

        if loss == 'mse':
            self.loss = nn.MSELoss()
        elif loss == 'l1':
            self.loss = nn.L1Loss()
        else:
            raise ValueError(f'Loss {loss} not recognised. Please use "mse" or "l1"')

        self.mean_val_metric_history = []
        self.mean_train_metric_history = []

        # dataset and dataloader
        dl = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=num_dl_workers)
        dl_val = DataLoader(dataset_val, batch_size=1, shuffle=False, pin_memory=True, num_workers=num_dl_workers)

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

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        loss_history = []

        # Save untrained metrics

        if self.accelerator.is_main_process:
            plot_samples(self.model, self.dl_val, '0', str(self.results_folder))
            self.evaluate_metrics()

        with tqdm(initial = self.epoch, total = self.train_num_epochs, disable=not accelerator.is_main_process) as pbar:

            while self.epoch < self.train_num_epochs:

                epoch_loss = []

                for data in self.dl:
                    data = data.to(device)

                    with self.accelerator.autocast():
                        pred = self.model(data)
                        loss = self.loss(pred, data)

                    epoch_loss.append(float(loss.item()))

                    self.accelerator.backward(loss)

                    pbar.set_description(f'loss: {loss.item():.4f}')

                    accelerator.wait_for_everyone()

                    self.opt.step()
                    self.opt.zero_grad()

                    accelerator.wait_for_everyone()

                self.epoch += 1
                epoch_loss = np.mean(epoch_loss)
                loss_history.append(epoch_loss)

                if accelerator.is_main_process:

                    if self.epoch % self.save_and_sample_every == 0:

                        logger.info(f'Saving model at epoch {self.epoch}')

                        self.save(self.epoch)
                        plot_samples(self.model, self.dl_val, f'{self.epoch}', str(self.results_folder))
                        self.evaluate_metrics()
                        self.write_loss_history(loss_history)

                pbar.update(1)

        if self.accelerator.is_main_process:
            self.write_loss_history(loss_history)
            self.evaluate_metrics()
            self.plot_metric_history()
            write_val_predictions(self.model, self.dl_val, 'final', str(self.results_folder))
            plot_val_prediction_slices(self.model, self.dl_val, self.results_folder)

        logger.info(f'[Accelerate device {self.accelerator.device}] Training complete!')

    def write_loss_history(self, loss_history):
        loss_history_path = self.results_folder / 'loss_history.json'
        with open(loss_history_path, 'w') as f:
            json.dump(loss_history, f)
        logger.info(f'Loss history written to {loss_history_path}')

        # Also write png loss plot
        fig, ax = plt.subplots(figsize=(3, 3), dpi=200)
        ax.plot(loss_history)
        ax.set_yscale('log')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        fig.tight_layout()
        fig.savefig(self.results_folder / 'loss_history.png')
        plt.close(fig)

    def evaluate_metrics(self):
        df_val = evaluate_autoencoder(self.model, self.dl_val, Path(self.results_folder) / 'val_metrics.csv', max_num_batches=len(self.dl_val), return_metrics=True)
        df_train = evaluate_autoencoder(self.model, self.dl, Path(self.results_folder) / 'train_metrics.csv', max_num_batches=len(self.dl_val), return_metrics=True)

        self.mean_val_metric_history.append((self.epoch, df_val.mean()))
        self.mean_train_metric_history.append((self.epoch, df_train.mean()))

    def plot_metric_history(self):
        epochs = [x[0] for x in self.mean_val_metric_history]
        df_val = pd.DataFrame([x[1] for x in self.mean_val_metric_history])
        df_train = pd.DataFrame([x[1] for x in self.mean_train_metric_history])

        fig, axs = plt.subplots(2, 3, figsize=(8, 6), dpi=200)

        for i, metric in enumerate(['MAE', 'MSE', 'Linf', 'SSIM', 'Dice', 'Hausdorff']):
            ax = axs.flatten()[i]
            ax.plot(epochs, df_val[metric], label='Validation')
            ax.plot(epochs, df_train[metric], label='Training')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.legend()

        fig.tight_layout()
        fig.savefig(self.results_folder / 'metric_history.png')
        plt.close(fig)


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def exists(x):
    return x is not None


def plot_val_prediction_slices(model, dl_val, results_folder: Path):
    outdir = results_folder / 'final_val_slice_plots'
    outdir.mkdir(exist_ok=True)

    for i, data in enumerate(dl_val):
        pred = model(data)
        pred = pred.detach().cpu().numpy().squeeze()
        data = data.detach().cpu().numpy().squeeze()

        fig, axs = plt.subplots(2, 3, figsize=(12, 8), dpi=200)

        for j in range(3):
            ax = axs[0, j]
            ax.set_title('Reconstructed')
            image = np.take(pred, indices=pred.shape[j] // 2, axis=j)
            im = ax.imshow(image, cmap="gray")
            fig.colorbar(im, ax=ax)

            ax = axs[1, j]
            ax.set_title('Original')
            image = np.take(data, indices=data.shape[j] // 2, axis=j)
            im = ax.imshow(image, cmap="gray")
            fig.colorbar(im, ax=ax)

        fig.tight_layout()
        filename = outdir / f"{i}.png"
        fig.savefig(filename)
        plt.close(fig)

        logger.info(f"Saved val spatial slice plot to {filename}")


def write_val_predictions(model, dl_val, name: str, results_folder: str):
    all_data_reconstructed = []
    all_data = []

    for i, data in enumerate(dl_val):
        pred = model(data)

        all_data_reconstructed.append(pred.detach().cpu().numpy().squeeze())
        all_data.append(data.detach().cpu().numpy().squeeze())

    all_data_reconstructed = np.stack(all_data_reconstructed, axis=0)
    all_data = np.stack(all_data, axis=0)

    np.savez_compressed(Path(results_folder) / f"{name}_val_predictions.npz", all_samples=all_data_reconstructed, all_data=all_data)
    logger.info(f'Saved val predictions to {results_folder}/{name}_val_predictions.npz')


def plot_samples(model, dl_val, name: str, results_folder: str, n_samples: int = 2):

    all_data_reconstructed = []
    all_data = []

    with torch.no_grad():
        for i, data in enumerate(dl_val):

            if i >= n_samples:
                break

            pred = model(data)

            all_data_reconstructed.append(pred.detach().cpu().numpy().squeeze())
            all_data.append(data.detach().cpu().numpy().squeeze())

    all_data_reconstructed = np.stack(all_data_reconstructed, axis=0)
    all_data = np.stack(all_data, axis=0)

    im_shape = all_data_reconstructed[0].shape
    logger.info(f'Sampled sequence shape: {im_shape}')

    # Now just visualise slices
    fig, axs = plt.subplots(n_samples, 4, figsize=(16, 3 * n_samples), dpi=200)

    for i in range(n_samples):
        for j in range(4):

                if j == 0:
                    ax = axs[i, j]
                    ax.set_title(f'Reconstructed (z={im_shape[2] // 2})')
                    im = ax.imshow(all_data_reconstructed[i][:, :, im_shape[2] // 2], cmap="gray")
                    fig.colorbar(im, ax=ax)

                elif j == 1:
                    ax = axs[i, j]
                    ax.set_title(f'Reconstructed (y={im_shape[1] // 2})')
                    im = ax.imshow(all_data_reconstructed[i][:, im_shape[1] // 2, :], cmap="gray")
                    fig.colorbar(im, ax=ax)

                elif j == 2:
                    ax = axs[i, j]
                    ax.set_title(f'Original (z={im_shape[2] // 2})')
                    im = ax.imshow(all_data[i][:, :, im_shape[2] // 2], cmap="gray")
                    fig.colorbar(im, ax=ax)

                elif j == 3:
                    ax = axs[i, j]
                    ax.set_title('Original (y={im_shape[1] // 2})')
                    im = ax.imshow(all_data[i][:, im_shape[1] // 2, :], cmap="gray")
                    fig.colorbar(im, ax=ax)

    outdir = Path(results_folder) / 'slice_plots'
    outdir.mkdir(exist_ok=True)

    fig.tight_layout()
    filename = outdir / f"{name}_samples_spatial_slice.png"
    fig.savefig(filename)
    plt.close(fig)

    logger.info(f"Saved samples spatial slice plot to {filename}")

    # Create isosurface plots
    isosurface_folder = Path(results_folder) / "isosurface_plots"
    isosurface_folder.mkdir(exist_ok=True)
    for i in range(n_samples):
        write_isosurface_plot_from_arr(all_data_reconstructed[i],
                                       dx=all_data_reconstructed[0].shape[-1],
                                       outname=isosurface_folder / f"{name}_{i}_reconstructed.png",
                                       level=0.5,
                                       verbose=False)

        write_isosurface_plot_from_arr(all_data[i],
                                        dx=all_data[0].shape[-1],
                                        outname=isosurface_folder / f"{name}_{i}_original.png",
                                        level=0.5,
                                        verbose=False)


