import json

import skimage
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction

from src.datasets.volumetric_datasets import VolumeDatasetInMemory
from conv_ae_3d.models.baseline_model import ConvAutoencoderBaseline

from src.interface_representation.interface_types import InterfaceType


def float_to_fraction(float_num):
    fraction = Fraction(float_num).limit_denominator()
    return f"{fraction.numerator}/{fraction.denominator}"


def plot_droplet_pdf_comparison(plot_label_to_data):
    fig, axs = plt.subplots(1, len(plot_label_to_data), figsize=(18, 6))

    for i, label in enumerate(plot_label_to_data):
        data = plot_label_to_data[label]
        axs[i].hist(data['gt'], bins=50, alpha=0.5, label='GT')
        axs[i].hist(data['pred'], bins=50, alpha=0.5, label='Pred')
        axs[i].set_title(label)
        axs[i].set_xlabel('Droplet size')
        axs[i].set_ylabel('Frequency')
        axs[i].legend()
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)


def read_loss_curve_from_file(outdir):
    loss_path = outdir / 'loss_history.json'
    with open(loss_path) as f:
        loss_curve = json.load(f)
    return loss_curve




def load_model(model_path, dim_mults=(1, 2, 4, 8, 8, 8), z_channels=4, block_type=1):

    # Defaults for these experiments
    dim = 32

    model = ConvAutoencoderBaseline(
        dim=dim,
        dim_mults=dim_mults,
        channels=1,
        z_channels=z_channels,
        block_type=block_type
    )

    device = torch.device('mps' if torch.mps.is_available() else 'cpu')
    print(f'Loading model to device: {device}')

    data = torch.load(str(model_path),
                      map_location=device)

    model.load_state_dict(data['model'])
    model.to(device)

    return model


def load_dataset(dataset_path, max_num_samples=None):
    return VolumeDatasetInMemory(
        data_dir=dataset_path,
        split='val',
        max_num_samples=max_num_samples
    )


def run_inference(dataset, model, N=None):
    gts = []
    preds = []

    device = torch.device('mps' if torch.mps.is_available() else 'cpu')
    print(f'Running inference on device: {device}')

    np.random.seed(0)
    inds = np.random.permutation(len(dataset))

    with torch.no_grad():
        for i in tqdm(inds[:N]):
            x = dataset[i].unsqueeze(0)
            x = x.to(device)
            y = model(x)

            gts.append(x.cpu().numpy().squeeze())
            preds.append(y.cpu().numpy().squeeze())

    return gts, preds


def compute_phi_sharp_from_tanh(phi):
    return np.heaviside(phi - 0.5, 1)


def compute_phi_sharp_from_sdf(psi):
    return np.heaviside(-psi, 1)


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

    gt_indices = np.argwhere(gt_patch) * 1 / 64
    pred_indices = np.argwhere(pred_patch) * 1 / 64

    if len(gt_indices) == 0 or len(pred_indices) == 0:
        return np.nan

    h_1 = directed_hausdorff(gt_indices, pred_indices)[0]
    h_2 = directed_hausdorff(pred_indices, gt_indices)[0]
    return max(h_1, h_2)


def get_phi_sharp(arr, interfacetype):
    if interfacetype == InterfaceType.TANH_EPSILON:
        return compute_phi_sharp_from_tanh(arr)
    elif (interfacetype == InterfaceType.SIGNED_DISTANCE_EXACT) or (interfacetype == InterfaceType.SIGNED_DISTANCE_APPROXIMATE):
        return compute_phi_sharp_from_sdf(arr)
    elif interfacetype == InterfaceType.HEAVISIDE:
        return compute_phi_sharp_from_tanh(arr)
    else:
        raise ValueError('Unknown interface type')


def get_phi_sharp_pred_and_gt(pred, gt, interface_type):
    if interface_type == InterfaceType.TANH_EPSILON:
        pred = compute_phi_sharp_from_tanh(pred)
        gt = compute_phi_sharp_from_tanh(gt)
    elif (interface_type == InterfaceType.SIGNED_DISTANCE_EXACT) or (interface_type == InterfaceType.SIGNED_DISTANCE_APPROXIMATE):
        pred = compute_phi_sharp_from_sdf(pred)
        gt = compute_phi_sharp_from_sdf(gt)
    elif interface_type == InterfaceType.HEAVISIDE:
        pred = compute_phi_sharp_from_tanh(pred)
        gt = compute_phi_sharp_from_tanh(gt)
    else:
        raise ValueError('Unknown interface type')
    return pred, gt


def order_samples_by_dice(outdir_to_metrics):
    outdir_to_ordered_samples = {}

    for outdir, metrics in outdir_to_metrics.items():
        ordered_samples = []

        dices = metrics['Dice']
        gts = metrics['gts']
        preds = metrics['preds']

        for dice, gt, pred in zip(dices, gts, preds):
            ordered_samples.append((dice, gt, pred))

        ordered_samples = sorted(ordered_samples, key=lambda x: x[0])
        outdir_to_ordered_samples[outdir] = ordered_samples

    return outdir_to_ordered_samples


def visualize_one_pred_gt_pair(gt, pred):
    # Use this to visualize good and bad examples. Show 3 3D meshes: pred, gt, and difference.
    difference = gt != pred
    difference = difference.astype(np.float32)

    fig = plt.figure(figsize=(12, 4), dpi=200)

    ax = fig.add_subplot(131, projection="3d")
    verts, faces, normals, values = skimage.measure.marching_cubes(
        gt, 0.5, spacing=(1, 1, 1), allow_degenerate=False, method='lewiner'
    )
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor("k")
    mesh.set_linewidth(0.05)
    mesh.set_alpha(0.9)
    ax.add_collection3d(mesh)
    ax.set_title('Ground Truth', fontsize=20)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)
    ax.set_zlim(0, 64)

    ax = fig.add_subplot(132, projection="3d")
    verts, faces, normals, values = skimage.measure.marching_cubes(
        pred, 0.5, spacing=(1, 1, 1), allow_degenerate=False, method='lewiner'
    )
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor("k")
    mesh.set_linewidth(0.05)
    mesh.set_alpha(0.9)
    ax.add_collection3d(mesh)
    ax.set_title('Prediction', fontsize=20)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)
    ax.set_zlim(0, 64)

    ax = fig.add_subplot(133, projection="3d")
    verts, faces, normals, values = skimage.measure.marching_cubes(
        difference, 0.5, spacing=(1, 1, 1), allow_degenerate=False, method='lewiner'
    )
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor("k")
    mesh.set_linewidth(0.05)
    mesh.set_alpha(0.9)
    ax.add_collection3d(mesh)
    ax.set_title('Error', fontsize=20)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)
    ax.set_zlim(0, 64)

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_comparison_of_instances(inds, outdir_to_metrics, outdir_to_params):
    outdirs = outdir_to_params.keys()

    for i in inds:
        for outdir in outdirs:
            gts = outdir_to_metrics[outdir]['gts']
            preds = outdir_to_metrics[outdir]['preds']
            interface_type = outdir_to_params[outdir]['interface_type']
            epsilon = outdir_to_params[outdir]['epsilon']

            gt = gts[i]
            pred = preds[i]
            dice = outdir_to_metrics[outdir]["Dice"][i]
            gt = get_phi_sharp(gt, interface_type)
            pred = get_phi_sharp(pred, interface_type)

            if np.sum(pred) > 0:
                print(interface_type, epsilon, dice)
                visualize_one_pred_gt_pair(gt, pred)
            else:
                print('Error visualising')


def plot_best_and_worst(n_best, n_worst, outdir_to_metrics, outdir_to_params, outdir):
    assert outdir in outdir_to_metrics
    assert outdir in outdir_to_params

    outdir_to_ordered_samples = order_samples_by_dice(outdir_to_metrics)

    for outdir, ordered_samples in outdir_to_ordered_samples.items():
        print(outdir)

        for i in range(n_best):
            print(f'Dice: {ordered_samples[-i - 1][0]}')
            gt = ordered_samples[-i - 1][1]
            pred = ordered_samples[-i - 1][2]
            gt = get_phi_sharp(gt, outdir_to_params[outdir]['interface_type'])
            pred = get_phi_sharp(pred, outdir_to_params[outdir]['interface_type'])
            print(gt.min(), gt.max())
            try:
                visualize_one_pred_gt_pair(gt, pred)
            except:
                print('Error visualizing')

        for i in range(n_worst):
            print(f'Dice: {ordered_samples[i][0]}')
            gt = ordered_samples[i][1]
            pred = ordered_samples[i][2]
            gt = get_phi_sharp(gt, outdir_to_params[outdir]['interface_type'])
            pred = get_phi_sharp(pred, outdir_to_params[outdir]['interface_type'])
            print(gt.min(), gt.max())
            try:
                visualize_one_pred_gt_pair(gt, pred)
            except:
                print('Error visualizing')
