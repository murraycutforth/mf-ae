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


def hausdorff_distance(gt_patch, pred_patch, level: float = 0.5, max_num_points: int = 100_000):
    """Returns the Hausdorff distance of the foreground region, obtained by thresholding the images at level

    Note:
        The distance is in units of voxels, assumes isotropic voxels

    Args:
        gt_patch: Ground truth patch
        pred_patch: Predicted patch
        level: Threshold level
        max_num_points: Maximum number of points to use in the distance calculation (for speed purposes)
    """
    gt_patch = gt_patch > level
    pred_patch = pred_patch > level

    gt_indices = np.argwhere(gt_patch)
    pred_indices = np.argwhere(pred_patch)

    if len(gt_indices) == 0 or len(pred_indices) == 0:
        return np.nan

    while len(gt_indices) > max_num_points:
        gt_indices = gt_indices[::2]
    while len(pred_indices) > max_num_points:
        pred_indices = pred_indices[::2]

    h_1 = directed_hausdorff(gt_indices, pred_indices)[0]
    h_2 = directed_hausdorff(pred_indices, gt_indices)[0]
    return max(h_1, h_2)


def evaluate_autoencoder(model, dataloader, outname: str, max_num_batches: int, return_metrics: bool = False) -> Optional[pd.DataFrame]:
    """Evaluate an autoencoder model on a dataset

    Assumes:
     - samples are 3D
     - data is single channel

    """
    metrics = []
    model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), desc='Running inference and computing metrics'):

            if i >= max_num_batches:
                break

            outputs = model(data).detach().cpu().numpy().squeeze()
            data = data.detach().cpu().numpy().squeeze()

            assert outputs.shape == data.shape, f"Output shape {outputs.shape} does not match data shape {data.shape}"

            if len(outputs.shape) == 4:
                # Iterate over batches as well
                batch_size = outputs.shape[0]
                for i in range(batch_size):
                    compute_metrics_single_array(data[i], metrics, outputs[i])

            else:
                compute_metrics_single_array(data, metrics, outputs)

    df = pd.DataFrame(metrics)

    logger.info(f'Computed all metrics for {outname}. Mean values: \n{df.mean()}')

    df.to_csv(outname)

    if return_metrics:
        return df


def compute_metrics_single_array(data, metrics, outputs):
    assert outputs.shape == data.shape
    assert len(outputs.shape) == 3

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


