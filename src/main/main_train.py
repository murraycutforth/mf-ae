import argparse
import logging

import torch.cuda
from torch import nn
from conv_ae_3d.models.baseline_model import ConvAutoencoderBaseline
from conv_ae_3d.trainer import MyAETrainer
from conv_ae_3d.metrics import MetricType

from src.paths import project_dir, local_data_dir
from src.datasets.baseline_dataset import PhiDataset

logger = logging.getLogger(__name__)


def main():
    args = parse_args()

    outdir = project_dir() / 'output' / args.run_name
    outdir.mkdir(exist_ok=True, parents=True)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(outdir / 'output.log')])

    logger.info(f'Starting training with args: {args}')

    model = construct_model(args)
    datasets = construct_datasets(args)
    loss = construct_loss(args)

    trainer = MyAETrainer(
        model=model,
        dataset_train=datasets['train'],
        dataset_val=datasets['val'],
        train_batch_size=args.batch_size,
        train_lr=args.lr,
        train_num_epochs=args.num_epochs,
        save_and_sample_every=args.save_and_sample_every,
        l2_reg=args.l2_reg,
        results_folder=outdir,
        cpu_only=not torch.cuda.is_available(),
        num_dl_workers=8,
        loss=loss,
        restart_from_milestone=args.restart_from_milestone,
        metric_types=[MetricType.MSE, MetricType.MAE, MetricType.LINF, MetricType.DICE, MetricType.HAUSDORFF],
    )

    trainer.train()


def construct_model(args):

    if args.activation == 'relu':
        activation = nn.ReLU()
    elif args.activation == 'leakyrelu':
        activation = nn.LeakyReLU()
    elif args.activation == 'selu':
        activation = nn.SELU()
    elif args.activation == 'elu':
        activation = nn.ELU()
    else:
        raise ValueError(f'Activation function {args.activation} not supported')

    if args.normalization == 'batch':
        norm = nn.BatchNorm3d
    elif args.normalization == 'instance':
        norm = nn.InstanceNorm3d
    else:
        raise ValueError(f'Normalization layer {args.normalization} not supported')

    model = ConvAutoencoderBaseline(
        image_shape=(256, 256, 256),
        activation=activation,
        norm=norm,
        feat_map_sizes=args.feat_map_sizes,
        linear_layer_sizes=args.linear_layer_sizes,
        final_activation='sigmoid',
    )

    return model


def construct_datasets(args) -> dict:
    return {
        'train': PhiDataset(data_dir=args.data_dir, split='train'),
        'val': PhiDataset(data_dir=args.data_dir, split='val'),
    }


def construct_loss(args):
    if args.loss == 'mse':
        return nn.MSELoss()
    elif args.loss == 'l1':
        return nn.L1Loss()
    else:
        raise ValueError(f'Loss function {args.loss} not supported')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default=local_data_dir(), help='Path to data directory')
    parser.add_argument('--run-name', type=str, default='debug', help='Name of the run')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=20, help='Number of epochs to train for')
    parser.add_argument('--save-and-sample-every', type=int, default=None, help='Save model and sample every n epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--feat-map-sizes', type=int, nargs='+', default=[4, 4, 4], help='Feature map sizes for the model')
    parser.add_argument('--linear-layer-sizes', type=int, nargs='+', default=None, help='Linear layer sizes in the bottleneck of the model')
    parser.add_argument('--loss', type=str, default='mse', help='Loss function to use')
    parser.add_argument('--l2-reg', type=float, default=0, help='L2 regularization strength')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function to use')
    parser.add_argument('--normalization', type=str, default='instance', help='Normalization layer to use')
    parser.add_argument('--restart-from-milestone', type=int, default=None, help='Restart training from a specific milestone')
    args = parser.parse_args()

    if args.save_and_sample_every is None:
        args.save_and_sample_every = args.num_epochs // 5

    return args


if __name__ == '__main__':
    main()
