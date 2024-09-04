"""In this script we conduct hyper-param tuning using Optuna.

Notes:
    - Currently hardcoded to my user (mcc4) on the yellowstone cluster.
    - This script assumes a cluster environment, where jobs are submitted via slurm.
    - The master process (this script) is expected to be executed on a single node, which submits jobs to the cluster.
    - The number of jobs to submit is specified by the n_jobs argument to study.optimize, the master process uses
    multi-threading parallelism to submit jobs to the cluster.
    - Jobs are submitted using subprocess, with a command constructed from the script name and the trial args.

"""

import logging
import subprocess
import time
from pathlib import Path
import tempfile

import pandas as pd
import optuna

from src.optuna_utils import finalise_study
from src.paths import project_dir

logger = logging.getLogger(__name__)

OUTDIR = project_dir() / 'output' / 'hyperparam_tuning'
OUTDIR.mkdir(exist_ok=True, parents=True)


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(OUTDIR / 'output.log')])

    logger.info('Starting hyperparameter tuning with Optuna')

    study = optuna.create_study(direction='maximize', study_name='debug', storage=f'sqlite:///{OUTDIR}/optuna.db')
    study.optimize(objective, n_trials=1, n_jobs=1, timeout=None)

    finalise_study(study, OUTDIR)


def objective(trial):
    trial_ind = trial.number

    jobscript = construct_bash_jobscript_yellowstone(trial)

    with tempfile.NamedTemporaryFile(mode='rw', delete=True, suffix='sh') as f:
        f.write(jobscript)
        f.flush()

        logger.info(f'Submitting job for trial {trial_ind}')
        logger.info(f'  Command: {jobscript}')

        output = subprocess.run(f'sbatch {f.name}', shell=True)

    assert output.returncode == 0, f'Job submission failed with return code {output.returncode}'

    job_outdir = Path(f'/home/darve/mcc4/mf-ae/output/trial_{trial_ind}')

    while not job_outdir.exists():
        time.sleep(10)

    logger.info(f'Job for trial {trial_ind} has started, output in {job_outdir}')

    while not (job_outdir / 'val_metrics_25.csv').exists():
        time.sleep(10)

    logger.info(f'Job for trial {trial_ind} has completed, output in {job_outdir}')

    val_metrics = pd.read_csv(job_outdir / 'val_metrics_25.csv')
    val_dice = val_metrics['dice'].mean()

    logger.info(f'Validation dice for trial {trial_ind}: {val_dice}')

    return val_dice


def construct_bash_jobscript_yellowstone(trial) -> str:

    lr = trial.suggest_float('lr', 1e-6, 1e-2, log=True)
    trial_ind = trial.number

    inner_cmd = f'''#!/bin/bash

#SBATCH -J trial_{trial_ind}               # Job name
#SBATCH -N 1                  # Total number of nodes requested
#SBATCH -t 12:05:00           # Run time (hh:mm:ss) - 5 minutes
#SBATCH -p gpu-pascal

source /home/darve/mcc4/codes/pytorch/pytorch_cuda-11.8/bin/activate

cd /home/darve/mcc4/mf-ae
python -m src.main.main_train --data-dir /home/darve/mcc4/data/multi_phase_droplet_data  --run-name trial_{trial_ind} --batch-size 1 --num-epochs 25 --save-and-sample-every 1000 --lr {lr} --feat-map-sizes 8 16 32 64 8
'''
    return inner_cmd


def construct_slurm_command(jobscript: str) -> str:
    return f'sbatch bash -c "{jobscript}"'


if __name__ == '__main__':
    main()
