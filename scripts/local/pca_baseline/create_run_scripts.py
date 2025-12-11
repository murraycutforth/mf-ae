import itertools
import os

VERSION = 'PCA'
DATA_DIR = '/Volumes/My\\ Passport\\ for\\ Mac/Multiphase-ae/preprocessed_datasets'


def main():
    search_args = {
        'data-dir': [
            DATA_DIR + '/spheres_mu_1.00/SIGNED_DISTANCE_EXACT',
            DATA_DIR + '/spheres_mu_1.00/HEAVISIDE',
            DATA_DIR + '/spheres_mu_1.00/TANH_EPSILON0.0078125',
            DATA_DIR + '/spheres_mu_1.00/TANH_EPSILON0.015625',
            DATA_DIR + '/spheres_mu_1.00/TANH_EPSILON0.03125',
            DATA_DIR + '/spheres_mu_1.00/TANH_EPSILON0.0625',
            DATA_DIR + '/spheres_mu_1.00/TANH_EPSILON0.125',
            DATA_DIR + '/spheres_mu_1.00/TANH_EPSILON0.25',
            DATA_DIR + '/spheres_mu_2.00/SIGNED_DISTANCE_EXACT',
            DATA_DIR + '/spheres_mu_2.00/HEAVISIDE',
            DATA_DIR + '/spheres_mu_2.00/TANH_EPSILON0.0078125',
            DATA_DIR + '/spheres_mu_2.00/TANH_EPSILON0.015625',
            DATA_DIR + '/spheres_mu_2.00/TANH_EPSILON0.03125',
            DATA_DIR + '/spheres_mu_2.00/TANH_EPSILON0.0625',
            DATA_DIR + '/spheres_mu_2.00/TANH_EPSILON0.125',
            DATA_DIR + '/spheres_mu_2.00/TANH_EPSILON0.25',
            DATA_DIR + '/spheres_mu_2.50/SIGNED_DISTANCE_EXACT',
            DATA_DIR + '/spheres_mu_2.50/HEAVISIDE',
            DATA_DIR + '/spheres_mu_2.50/TANH_EPSILON0.0078125',
            DATA_DIR + '/spheres_mu_2.50/TANH_EPSILON0.015625',
            DATA_DIR + '/spheres_mu_2.50/TANH_EPSILON0.03125',
            DATA_DIR + '/spheres_mu_2.50/TANH_EPSILON0.0625',
            DATA_DIR + '/spheres_mu_2.50/TANH_EPSILON0.125',
            DATA_DIR + '/spheres_mu_2.50/TANH_EPSILON0.25',
        ],
        'seed': [4]
    }

    const_args = {
        'lr': 1e-5,
        'model-type': 'pca',
        'dataset-type': 'volumetric',
        'num-dl-workers': 0,
        'batch-size': 32,
        'num-epochs': 25,
        'loss': 'mse',
    }

    # Generate the Cartesian product of the values in search_args
    keys, values = zip(*search_args.items())
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]

    # Iterate over the combinations
    for i, combination in enumerate(combinations):
        run_name = create_param_str(i, combination)
        args_dict = {**const_args, **combination}
        create_run_script(i, run_name, args_dict)

    create_run_all_script(len(combinations))


def create_param_str(i, combination):
    param_str = "_".join([f"{key}{str(value)}" for key, value in combination.items()])
    param_str = param_str.replace('/', '_')
    param_str = param_str.replace("(", "")
    param_str = param_str.replace(")", "")
    param_str = param_str.replace(" ", "")
    param_str = param_str.replace(",", "")
    param_str = param_str.replace(".", "")
    param_str = param_str.replace("-", "")

    run_name = f"interfacial_ae_v{VERSION}_run_{i:02}_" + param_str
    print(run_name)
    return run_name



def create_run_script(i, run_name, args_dict):
    args_dict['run-name'] = run_name

    # Create .sh
    with open(f"run_training_v{VERSION}_{i}.sh", "w") as f:
        f.write("export PYTHONPATH=/Users/murraycutforth/Projects/mf-ae\n")

        param_str = ""
        for k, v in args_dict.items():
            param_str += f" --{k} {v}"

        f.write(f"python /Users/murraycutforth/Projects/mf-ae/src/main/main_train.py {param_str}\n")

    os.chmod(f"run_training_v{VERSION}_{i}.sh", 0o755)

    print(f"Created run script for combination {i}")



def create_run_all_script(num_runs: int):
    with open('run_all.sh', 'w') as f:
        f.write('#!/bin/bash\n')
        for i in range(num_runs):
            f.write(f'./run_training_v{VERSION}_{i}.sh\n')
            f.write('sleep 0.5\n')
        f.write('echo "All jobs submitted"')

    os.chmod("run_all.sh", 0o755)


if __name__ == '__main__':
    main()
