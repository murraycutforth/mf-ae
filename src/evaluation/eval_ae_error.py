import json

import torch
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import pandas as pd


def l1_error(gt_patch, pred_patch):
    return np.linalg.norm(gt_patch.flatten() - pred_patch.flatten(), ord=1) / gt_patch.size


def l2_error(gt_patch, pred_patch):
    return np.linalg.norm(gt_patch.flatten() - pred_patch.flatten(), ord=2) / gt_patch.size


def linf_error(gt_patch, pred_patch):
    return np.linalg.norm(gt_patch.flatten() - pred_patch.flatten(), ord=np.inf) / gt_patch.size


def ssim_error(gt_patch, pred_patch):
    return ssim(gt_patch, pred_patch, data_range=gt_patch.max() - gt_patch.min())


def evaluate_autoencoder(model, dataloader, outdir) -> None:
    """Evaluate an autoencoder model on a dataset

    Assumes:
     - dataloader has batch size 1
     - samples are 3D
     - data is single channel

    """
    metrics = []
    model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader, desc='Evaluating'):
            outputs = model(data).numpy().squeeze()
            data = data.numpy().squeeze()

            assert outputs.shape == data.shape
            assert len(outputs.shape) == 3

            l1 = l1_error(data, outputs)
            l2 = l2_error(data, outputs)
            linf = linf_error(data, outputs)
            ssim_score = ssim_error(data, outputs)

            metrics.append({
                'L1': l1,
                'L2': l2,
                'Linf': linf,
                'SSIM': ssim_score
            })

    df = pd.DataFrame(metrics)
    df.to_csv(outdir / 'metrics.csv')

