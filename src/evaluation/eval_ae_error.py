from pathlib import Path
from typing import Optional
import logging

import torch
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import directed_hausdorff
import pandas as pd

logger = logging.getLogger(__name__)


def compute_mae(gt_patch, pred_patch):
    return np.linalg.norm(gt_patch.flatten() - pred_patch.flatten(), ord=1) / gt_patch.size


def compute_mse(gt_patch, pred_patch):
    return (np.linalg.norm(gt_patch.flatten() - pred_patch.flatten(), ord=2))**2 / gt_patch.size


def linf_error(gt_patch, pred_patch):
    return np.linalg.norm(gt_patch.flatten() - pred_patch.flatten(), ord=np.inf)


def ssim_error(gt_patch, pred_patch):
    return ssim(gt_patch, pred_patch, data_range=gt_patch.max() - gt_patch.min())


def dice_coefficient(gt_patch, pred_patch, level: float = 0.5):
    """Returns the dice coefficient of foreground region, obtained by thresholding the images at level
    """
    gt_patch = gt_patch > level
    pred_patch = pred_patch > level
    intersection = np.sum(gt_patch * pred_patch)
    union = np.sum(gt_patch) + np.sum(pred_patch)
    return 2 * intersection / union


def hausdorff_distance(gt_patch, pred_patch, level: float = 0.5):
    """Returns the Hausdorff distance of the foreground region, obtained by thresholding the images at level
    """
    gt_patch = gt_patch > level
    pred_patch = pred_patch > level

    gt_indices = np.argwhere(gt_patch)
    pred_indices = np.argwhere(pred_patch)

    if len(gt_indices) == 0 or len(pred_indices) == 0:
        return np.nan

    # Note- at this point if we wanted to apply a scale factor to the distance, we could do so here
    # As it stands, the HD is in units of voxel length, assumes isotropic voxels
    # For speed purposes, compute on downsampled point clouds if they are too large

    while len(gt_indices) > 100_000:
        gt_indices = gt_indices[::2]
    while len(pred_indices) > 100_000:
        pred_indices = pred_indices[::2]

    h_1 = directed_hausdorff(gt_indices, pred_indices)[0]
    h_2 = directed_hausdorff(pred_indices, gt_indices)[0]
    return max(h_1, h_2)


def evaluate_autoencoder(model, dataloader, outname: Path, return_metrics: bool = False) -> Optional[pd.DataFrame]:
    """Evaluate an autoencoder model on a dataset

    Assumes:
     - dataloader has batch size 1
     - samples are 3D
     - data is single channel

    """
    metrics = []
    model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader, desc='Running inference and computing metrics'):
            outputs = model(data).detach().cpu().numpy().squeeze()
            data = data.detach().cpu().numpy().squeeze()

            assert outputs.shape == data.shape, f"Output shape {outputs.shape} does not match data shape {data.shape}"
            assert len(outputs.shape) == 3, f"Output shape {outputs.shape} is not 3D"

            mae = compute_mae(data, outputs)
            mse = compute_mse(data, outputs)
            linf = linf_error(data, outputs)
            ssim_score = ssim_error(data, outputs)
            dice = dice_coefficient(data, outputs, level=0.5)
            hausdorff = hausdorff_distance(data, outputs, level=0.5)

            metrics.append({
                'MAE': mae,
                'MSE': mse,
                'Linf': linf,
                'SSIM': ssim_score,
                'Dice': dice,
                'Hausdorff': hausdorff,
            })

    df = pd.DataFrame(metrics)

    logger.info(f'Computed all metrics for {outname.stem}. Mean values: {df.mean()}')

    df.to_csv(outname)

    if return_metrics:
        return df


