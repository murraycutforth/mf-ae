import logging
from pathlib import Path

import torch.cuda
from conv_ae_3d.trainer import MyAETrainer
from conv_ae_3d.metrics import MetricType

from src.main.main_train import parse_args, construct_model, construct_datasets

logger = logging.getLogger(__name__)


def main():
    args = parse_args()

    assert args.restart_dir is not None, 'Need to specify restart_dir'
    assert args.restart_from_milestone is not None, 'Need to specify restart_from_milestone'

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(outdir / 'output.log')])

    logger.info(f'Starting eval with args: {args}')

    model = construct_model(args)
    datasets = construct_datasets(args)

    trainer = MyAETrainer(
        model=model,
        dataset_train=datasets['train'],
        dataset_val=datasets['val'],
        train_batch_size=args.batch_size,
        train_lr=args.lr,
        train_num_epochs=args.num_epochs,
        results_folder=str(outdir),
        cpu_only=not torch.cuda.is_available(),
        num_dl_workers=8,
        restart_from_milestone=args.restart_from_milestone,
        restart_dir=args.restart_dir,
        metric_types=[MetricType.MSE, MetricType.MAE, MetricType.LINF, MetricType.DICE, MetricType.HAUSDORFF],
    )

    trainer.eval()


if __name__ == '__main__':
    main()
