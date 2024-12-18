import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm
from conv_ae_3d.metrics import compute_mae, compute_mse, linf_error, ssim_error, dice_coefficient

from src.paths import project_dir, local_data_dir
from src.plotting_utils import write_isosurface_plot_from_arr
from src.datasets.phi_field_dataset import PhiDataset

logger = logging.getLogger(__name__)


def main():
    data_dir = project_dir() / 'output' / 'PCA'
    data_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f'Using data directory: {data_dir}')

    for n_components in [1, 2, 3, 4, 5, 10, 20, 50]:
        outdir = data_dir / f'n_components_{n_components}'
        outdir.mkdir(exist_ok=True, parents=True)
        run_pca(n_components, outdir)


def run_pca(n_components, outdir):
    logger.info(f'Running PCA with {n_components} components')

    model = IncrementalPCA(n_components=n_components)
    datasets = construct_datasets()
    batch_size = 50
    N_train_batches = len(datasets['train']) // batch_size

    for i in tqdm(range(N_train_batches)):
        i_start = i * batch_size
        i_end = min((i + 1) * batch_size, len(datasets['train']))
        batch = [datasets['train'][i].squeeze().numpy() for i in range(i_start, i_end)]
        batch = np.array(batch).reshape(len(batch), -1)
        model.partial_fit(batch)

    logger.info('PCA fit complete')

    evaluate_pca(model, datasets['val'], outdir / 'metrics.csv')
    plot_pca(model, datasets['val'], outdir)


def plot_pca(model, dataset, outdir):

    isosurface_folder = Path(outdir) / "isosurface_plots"
    isosurface_folder.mkdir(exist_ok=True)
    n_samples = 10

    all_data_reconstructed = []
    all_data = []

    for i, data in enumerate(dataset):
        if i >= n_samples:
            break

        data = data.numpy().squeeze()

        image_shape = data.shape
        data = data.reshape(1, -1)

        z = model.transform(data)
        outputs = model.inverse_transform(z)

        outputs = outputs.reshape(image_shape)
        data = data.reshape(image_shape)

        assert outputs.shape == data.shape
        assert len(outputs.shape) == 3

        all_data_reconstructed.append(outputs)
        all_data.append(data)

    # Create isosurface plots for first 10 samples
    for i in range(min(10, len(all_data_reconstructed))):
        write_isosurface_plot_from_arr(all_data_reconstructed[i],
                                       dx=all_data_reconstructed[0].shape[-1],
                                       outname=isosurface_folder / f"pca_{i}_reconstructed.png",
                                       level=0.5,
                                       verbose=False)

        write_isosurface_plot_from_arr(all_data[i],
                                       dx=all_data[0].shape[-1],
                                       outname=isosurface_folder / f"pca_{i}_original.png",
                                       level=0.5,
                                       verbose=False)


def construct_datasets() -> dict:
    data_dir = local_data_dir()

    return {
        'train': PhiDataset(data_dir=data_dir, split='train'),  # Returns tensors of shape (1, 256, 256, 256)
        'val': PhiDataset(data_dir=data_dir, split='val'),
    }


def evaluate_pca(model, dataset, outname, return_metrics: bool = False) -> Optional[pd.DataFrame]:
    """Evaluate an autoencoder model on a dataset

    Assumes:
     - dataloader has batch size 1
     - samples are 3D
     - data is single channel

    """
    metrics = []

    for data in dataset:
        data = data.numpy().squeeze()

        image_shape = data.shape
        data = data.reshape(1, -1)

        z = model.transform(data)
        outputs = model.inverse_transform(z)

        outputs = outputs.reshape(image_shape)
        data = data.reshape(image_shape)

        assert outputs.shape == data.shape
        assert len(outputs.shape) == 3

        mae = compute_mae(data, outputs)
        mse = compute_mse(data, outputs)
        linf = linf_error(data, outputs)
        ssim_score = ssim_error(data, outputs)
        dice = dice_coefficient(data, outputs, level=0.5)

        metrics.append({
            'MAE': mae,
            'MSE': mse,
            'Linf': linf,
            'SSIM': ssim_score,
            'Dice': dice
        })

    df = pd.DataFrame(metrics)

    if return_metrics:
        return df
    else:
        df.to_csv(outname)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
