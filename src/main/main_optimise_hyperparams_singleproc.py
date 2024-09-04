"""In this script """

import logging
import argparse

import matplotlib.pyplot as plt
import torch.cuda
from torch import nn
import optuna
from optuna.visualization.matplotlib import plot_contour
from optuna.visualization.matplotlib import plot_edf
from optuna.visualization.matplotlib import plot_intermediate_values
from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_parallel_coordinate
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_rank
from optuna.visualization.matplotlib import plot_slice
from optuna.visualization.matplotlib import plot_timeline
from conv_ae_3d.trainer import MyAETrainer
from conv_ae_3d.models.baseline_model import ConvAutoencoderBaseline
from conv_ae_3d.metrics import MetricType

from src.optuna_utils import finalise_study
from src.paths import project_dir, local_data_dir
from src.datasets.baseline_dataset import PhiDataset

logger = logging.getLogger(__name__)

OUTDIR = project_dir() / 'output' / 'hyperparam_tuning'
OUTDIR.mkdir(exist_ok=True, parents=True)


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(OUTDIR / 'output.log')])

    logger.info('Starting hyperparameter tuning with Optuna')

    study = optuna.create_study(direction='maximize', study_name='debug', storage=f'sqlite:///{OUTDIR}/optuna.db')
    study.optimize(objective, n_trials=5, n_jobs=1, timeout=None)

    finalise_study(study, OUTDIR)


def objective(trial):

    activation_str = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'SELU', 'ELU'])
    norm_str = trial.suggest_categorical('norm', ['BatchNorm3d', 'InstanceNorm3d'])
    feat_map_sizes_str = trial.suggest_categorical('feat_map_sizes', ['4_4_4_4', '2_2_2_2'])
    linear_layer_sizes_str = trial.suggest_categorical('linear_layer_sizes', ['None', '100', '1000_500_100'])
    loss_str = trial.suggest_categorical('loss', ['mse', 'l1'])

    with SuppressLogging():

        activation = getattr(nn, activation_str)()
        norm = getattr(nn, norm_str)
        feat_map_sizes = tuple(map(int, feat_map_sizes_str.split('_')))
        linear_layer_sizes = None if linear_layer_sizes_str == 'None' else tuple(
            map(int, linear_layer_sizes_str.split('_')))
        loss = construct_loss(loss_str)

        model = ConvAutoencoderBaseline(
            image_shape=(256, 256, 256),
            activation=activation,
            norm=norm,
            feat_map_sizes=feat_map_sizes,
            linear_layer_sizes=linear_layer_sizes,
            final_activation='sigmoid',
        )

        datasets = construct_datasets()

        trainer = MyAETrainer(
            model=model,
            dataset_train=datasets['train'],
            dataset_val=datasets['val'],
            train_batch_size=1,
            train_lr=trial.suggest_float('lr', 1e-6, 1e-2, log=True),
            train_num_epochs=3,
            save_and_sample_every=51,
            results_folder=None,
            l2_reg=trial.suggest_float('l2_reg', 1e-6, 1e-2, log=True),
            cpu_only=not torch.cuda.is_available(),
            num_dl_workers=0,
            loss=loss,
            metric_types=[MetricType.DICE],
        )

        trainer.train()

        results = trainer.mean_val_metrics

    return results['DICE']


def construct_datasets() -> dict:
    args = parse_args()

    return {
        'train': PhiDataset(data_dir=args.data_dir, split='train', debug=True),
        'val': PhiDataset(data_dir=args.data_dir, split='val', debug=True),
    }


def construct_loss(loss_type: str) -> nn.Module:
    if loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'l1':
        return nn.L1Loss()
    else:
        raise ValueError(f'Loss function {loss_type} not supported')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default=local_data_dir(), help='Path to data directory')
    return parser.parse_args()


class SuppressLogging:
    """Context manager to suppress logging output
    """
    def __enter__(self):
        self._original_level = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.CRITICAL)

    def __exit__(self, exc_type, exc_value, traceback):
        logging.getLogger().setLevel(self._original_level)


if __name__ == '__main__':
    main()
