import json
import pathlib

from src.interface_representation.interface_types import InterfaceType


def read_loss_curve_from_file(outdir):
    loss_path = outdir / 'loss_history.json'
    with open(loss_path) as f:
        loss_curve = json.load(f)
    return loss_curve


def extract_compression_ratio(outdir):
    run_info_path = outdir / 'run_info.json'

    with open(run_info_path) as f:
        run_info = json.load(f)

    return run_info['compression_ratio']


def extract_interface_type(outdir):
    final_part = '_'.join(outdir.stem.split('_')[5:-2])
    final_part = final_part.replace('datadir', '')

    str_to_type = {
        'TANH_EPSILON00078125': InterfaceType.TANH_EPSILON,
        'TANH_EPSILON0015625': InterfaceType.TANH_EPSILON,
        'TANH_EPSILON003125': InterfaceType.TANH_EPSILON,
        'TANH_EPSILON00625': InterfaceType.TANH_EPSILON,
        'TANH_EPSILON0125': InterfaceType.TANH_EPSILON,
        'TANH_EPSILON025': InterfaceType.TANH_EPSILON,
        'HEAVISIDE': InterfaceType.HEAVISIDE,
        'SIGNED_DISTANCE_EXACT': InterfaceType.SIGNED_DISTANCE_EXACT,
        'SIGNED_DISTANCE_APPROXIMATE': InterfaceType.SIGNED_DISTANCE_APPROXIMATE,
    }

    return str_to_type[final_part]


def extract_epsilon(outdir):
    final_part = '_'.join(outdir.stem.split('_')[5:-2])

    # Remove 'datadir' from start of string
    final_part = final_part.replace('datadir', '')

    str_to_epsilon = {
        'TANH_EPSILON00078125': 0.0078125,
        'TANH_EPSILON0015625': 0.015625,
        'TANH_EPSILON003125': 0.03125,
        'TANH_EPSILON00625': 0.0625,
        'TANH_EPSILON0125': 0.125,
        'TANH_EPSILON025': 0.25,
        'HEAVISIDE': None,
        'SIGNED_DISTANCE_EXACT': None,
        'SIGNED_DISTANCE_APPROXIMATE': None,
    }

    return str_to_epsilon[final_part]


def get_model_path(outdir, epoch: int = 20):
    return outdir / f'model-{epoch}.pt'


def get_dim_mults(outdir):
    model_args_path = outdir / 'construct_model_args.json'

    with open(model_args_path) as f:
        model_args = json.load(f)

    return model_args['dim_mults']


def get_max_train_samples(outdir):
    final_part = outdir.stem.split('_')[-1]
    final_part = final_part.replace('maxtrainsamples', '')
    max_train_samples = int(final_part)
    return max_train_samples


def get_dataset_path(interfacetype, epsilon):
    if interfacetype == InterfaceType.TANH_EPSILON:
        return pathlib.Path(
            f'/Users/murray/Projects/multphase_flow_encoder/multiphase_flow_encoder/src/preprocessing/data/patched_hit_experiment/TANH_EPSILON{epsilon}')
    elif interfacetype == InterfaceType.HEAVISIDE:
        return pathlib.Path(f'/src/preprocessing/data/patched_hit_experiment/HEAVISIDE')
    elif interfacetype == InterfaceType.SIGNED_DISTANCE_EXACT:
        return pathlib.Path(f'/src/preprocessing/data/patched_hit_experiment/SIGNED_DISTANCE_EXACT')
    elif interfacetype == InterfaceType.SIGNED_DISTANCE_APPROXIMATE:
        return pathlib.Path(f'/src/preprocessing/data/patched_hit_experiment/SIGNED_DISTANCE_APPROXIMATE')
    else:
        raise ValueError('Unknown interface type')

