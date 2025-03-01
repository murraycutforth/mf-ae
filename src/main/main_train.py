import argparse
import json
import logging
import sys

import torch.cuda
from conv_ae_3d.models.conv_with_fc_model import ConvAutoencoderWithFC
from torch import nn
from conv_ae_3d.models.baseline_model import ConvAutoencoderBaseline
from conv_ae_3d.trainer_ae import MyAETrainer
from conv_ae_3d.metrics import MetricType

from src.paths import project_dir
from src.datasets.phi_field_dataset import PhiDataset, PatchPhiDataset
from src.datasets.ellipse_dataset import EllipseDataset
from src.datasets.spheres_dataset import SpheresDataset
from src.datasets.volumetric_datasets import VolumeDatasetInMemory, PatchVolumeDatasetInMemory

logger = logging.getLogger(__name__)


def main():
    args = parse_args()
    outdir = project_dir() / 'output' / args.run_name
    outdir.mkdir(exist_ok=True, parents=True)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(outdir / 'output.log')])

    logger.info(f'Starting training with args: {args}')
    logger.info(f'Using output directory: {outdir}')

    save_args_to_file(args, outdir / 'args.json')

    datasets = construct_datasets(args)
    model = construct_model(args, outdir)
    loss = construct_loss(args)

    trainer = MyAETrainer(
        model=model,
        dataset_train=datasets['train'],
        dataset_val=datasets['val'],
        train_batch_size=args.batch_size,
        train_lr=args.lr,
        train_num_epochs=args.num_epochs,
        l2_reg=args.weight_decay,
        save_and_sample_every=args.save_and_sample_every,
        results_folder=outdir,
        cpu_only=not torch.cuda.is_available(),
        num_dl_workers=args.num_dl_workers,
        loss=loss,
        metric_types=[MetricType.MSE, MetricType.MAE, MetricType.LINF],
        restart_from_milestone=args.restart_from_milestone,
        restart_dir=args.restart_dir,
    )

    trainer.train()


def main():
    args = parse_args()

    outdir = project_dir() / 'output' / args.run_name
    outdir.mkdir(exist_ok=True, parents=True)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(outdir / 'output.log')])

    logger.info(f'Starting training with args: {args}')

    model = construct_model(args, outdir=outdir)
    datasets = construct_datasets(args)
    loss = construct_loss(args)
    metrics = construct_metrics(args)

    trainer = MyAETrainer(
        model=model,
        dataset_train=datasets['train'],
        dataset_val=datasets['val'],
        train_batch_size=args.batch_size,
        train_lr=args.lr,
        train_num_epochs=args.num_epochs,
        results_folder=outdir,
        l2_reg=args.weight_decay,
        cpu_only=not torch.cuda.is_available(),
        num_dl_workers=args.num_dl_workers,
        loss=loss,
        restart_dir=args.restart_dir,
        restart_from_milestone=args.restart_from_milestone,
        metric_types=metrics,
        low_data_mode=True,
    )

    trainer.train()


def construct_model(args, outdir=None):

    torch.manual_seed(args.seed)

    if args.model_type == 'baseline':
        model = ConvAutoencoderBaseline(
            dim=args.dim,
            dim_mults=args.dim_mults,
            channels=1,
            z_channels=args.z_channels,
            block_type=args.block_type,
        )
    elif args.model_type == 'conv_with_fc':
        model = ConvAutoencoderWithFC(
            dim=args.dim,
            dim_mults=args.dim_mults,
            channels=1,
            z_channels=args.z_channels,
            block_type=args.block_type,
            fc_layers=args.fc_layers,
            image_shape=(64, 64, 64)
        )
    else:
        raise ValueError(f'Model type {args.model_type} not supported')

    if outdir is not None:
        with open(outdir / 'construct_model_args.json', 'w') as f:
            json.dump(vars(args), f)

    return model


def construct_datasets(args) -> dict:
    if args.dataset_type == 'volumetric':
        return {
            'train': VolumeDatasetInMemory(debug=args.debug,
                                           data_dir=args.data_dir,
                                           split='train',
                                           max_num_samples=args.max_samples,
                                           ),
            'val': VolumeDatasetInMemory(debug=args.debug,
                                         data_dir=args.data_dir,
                                         split='val',
                                            max_num_samples=args.max_samples,
                                         )
        }
    elif args.dataset_type == 'volumetric_patched':
        return {
            'train': PatchVolumeDatasetInMemory(debug=args.debug,
                                                data_dir=args.data_dir,
                                                split='train',
                                                patch_size=args.vol_size,
                                                dtype=torch.float16),
            'val': PatchVolumeDatasetInMemory(debug=args.debug,
                                              data_dir=args.data_dir,
                                              split='val',
                                              patch_size=args.vol_size,
                                              dtype=torch.float16),
        }
    else:
        raise ValueError(f'Dataset type {args.dataset_type} not supported')


def construct_loss(args):
    if args.loss == 'mse':
        return nn.MSELoss()
    elif args.loss == 'l1':
        return nn.L1Loss()
    else:
        raise ValueError(f'Loss function {args.loss} not supported')


def construct_metrics(args) -> list[MetricType]:
    metrics = []
    for metric in args.metrics:
        if metric == 'mse':
            metrics.append(MetricType.MSE)
        elif metric == 'mae':
            metrics.append(MetricType.MAE)
        elif metric == 'linf':
            metrics.append(MetricType.LINF)
        elif metric == 'sdf_heaviside':
            metrics.append(MetricType.SDF_HEAVISIDE_L1)
        elif metric == 'tanh_heaviside':
            metrics.append(MetricType.TANH_HEAVISIDE_L1)
        else:
            raise ValueError(f'Metric {metric} not supported')
    return metrics



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', type=str, default='debug', help='Name of the run')

    # Dataset args
    parser.add_argument('--dataset-type', type=str, default='volumetric', help='Type of dataset to use')
    parser.add_argument('--data-dir', type=str, default='/Users/murray/Projects/multphase_flow_encoder/multiphase_flow_encoder/src/preprocessing/data/mu_spheres/spheres_mu_1.00/HEAVISIDE', help='Path to data directory')
    parser.add_argument('--num-dl-workers', type=int, default=0, help='Number of dataloader workers')
    parser.add_argument('--debug', action='store_true', help='Debug mode - run with just a few data samples')
    parser.add_argument('--vol-size', type=int, default=64, help='Size of volumes / patches to use')
    parser.add_argument('--max-samples', type=int, default=None, help='Maximum number of samples to use')

    # Training args
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=10, help='Number of epochs to train for')
    parser.add_argument('--save-and-sample-every', type=int, default=None, help='Save and sample every n epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate for training')
    parser.add_argument('--loss', type=str, default='mse', help='Loss function to use')
    parser.add_argument('--weight-decay', type=float, default=1e-6, help='Weight decay for optimizer')
    parser.add_argument('--restart-from-milestone', type=int, default=None, help='Restart training from this milestone (epoch number)')
    parser.add_argument('--restart-dir', type=str, default=None, help='Directory to restart training from')
    parser.add_argument('--metrics', type=str, nargs='+', default=['mse', 'mae', 'linf'], help='Metrics to compute during training')

    # Model args
    parser.add_argument('--model-type', type=str, default='baseline', help='Type of model to use (baseline, conv_with_fc)')
    parser.add_argument('--dim', type=int, default=8, help='Base dimension for the model')
    parser.add_argument('--dim-mults', type=int, nargs='+', default=[1, 2,], help='Dimension multipliers for the model')
    parser.add_argument('--fc-layers', type=int, nargs='+', default=None, help='Fully connected layers to use in the model')
    parser.add_argument('--z-channels', type=int, default=1, help='Number of channels in the latent space')
    parser.add_argument('--block-type', type=int, default=1, help='Type of block to use in the model')

    args = parser.parse_args()

    args.dim_mults = tuple(args.dim_mults)

    return args


def save_args_to_file(args, filepath):
    with open(filepath, 'w') as f:
        json.dump(vars(args), f, indent=4)


if __name__ == '__main__':
    main()
