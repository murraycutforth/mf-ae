import argparse
import logging

import torch.cuda
from torch import nn

from src.paths import project_dir, local_data_dir
from src.trainer.trainer import MyAETrainer
from src.models.conv_ae import ConvAutoencoderBaseline
from src.evaluation.eval_ae_error import evaluate_autoencoder
from src.datasets.baseline_dataset import PhiDataset

logger = logging.getLogger(__name__)


def main():
    args = parse_args()
    logger.info(f'Using data directory: {args.data_dir}')

    outdir = project_dir() / 'output' / args.run_name
    outdir.mkdir(exist_ok=True)

    model = construct_model(args)
    datasets = construct_datasets(args)

    trainer = MyAETrainer(
        model=model,
        dataset=datasets['train'],
        dataset_val=datasets['val'],
        train_batch_size=args.batch_size,
        train_lr=3e-4,
        train_num_epochs=args.num_epochs,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=10,
        results_folder=outdir,
        amp=False,
        mixed_precision_type='fp16',
        cpu_only=not torch.cuda.is_available(),
    )

    trainer.train()

    evaluate_autoencoder(trainer.model, trainer.dataset_val, outdir)



def construct_model(args):
    model = ConvAutoencoderBaseline(
        image_shape=(256, 256, 256),
        flat_bottleneck=False,
        latent_dim=100,
        activation=nn.SELU(),
        norm=nn.InstanceNorm3d,
        feat_map_sizes=(16, 32, 64)
    )

    return model


def construct_datasets(args) -> dict:
    return {
        'train': PhiDataset(data_dir=args.data_dir, split='train'),
        'val': PhiDataset(data_dir=args.data_dir, split='val'),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default=local_data_dir(), help='Path to data directory')
    parser.add_argument('--run-name', type=str, default='debug', help='Name of the run')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of epochs to train for')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()