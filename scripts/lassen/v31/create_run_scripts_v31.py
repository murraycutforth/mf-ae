import itertools
import os

VERSION = 31  # Compare effect of training set size


def main():
    search_args = {
        'data-dir': ['/usr/workspace/cutforth1/data-mf-ae/spheres_mu_1.00/SIGNED_DISTANCE_EXACT',
                     '/usr/workspace/cutforth1/data-mf-ae/spheres_mu_1.00/HEAVISIDE',
                     '/usr/workspace/cutforth1/data-mf-ae/spheres_mu_1.00/TANH_EPSILON0.0078125',
                     '/usr/workspace/cutforth1/data-mf-ae/spheres_mu_1.00/TANH_EPSILON0.015625',
                     '/usr/workspace/cutforth1/data-mf-ae/spheres_mu_1.00/TANH_EPSILON0.03125',
                     '/usr/workspace/cutforth1/data-mf-ae/spheres_mu_1.00/TANH_EPSILON0.0625',
                     '/usr/workspace/cutforth1/data-mf-ae/spheres_mu_1.00/TANH_EPSILON0.125',
                     '/usr/workspace/cutforth1/data-mf-ae/spheres_mu_1.00/TANH_EPSILON0.25',
                     '/usr/workspace/cutforth1/data-mf-ae/spheres_mu_2.00/SIGNED_DISTANCE_EXACT',
                     '/usr/workspace/cutforth1/data-mf-ae/spheres_mu_2.00/HEAVISIDE',
                     '/usr/workspace/cutforth1/data-mf-ae/spheres_mu_2.00/TANH_EPSILON0.0078125',
                     '/usr/workspace/cutforth1/data-mf-ae/spheres_mu_2.00/TANH_EPSILON0.015625',
                     '/usr/workspace/cutforth1/data-mf-ae/spheres_mu_2.00/TANH_EPSILON0.03125',
                     '/usr/workspace/cutforth1/data-mf-ae/spheres_mu_2.00/TANH_EPSILON0.0625',
                     '/usr/workspace/cutforth1/data-mf-ae/spheres_mu_2.00/TANH_EPSILON0.125',
                     '/usr/workspace/cutforth1/data-mf-ae/spheres_mu_2.00/TANH_EPSILON0.25',
                     '/usr/workspace/cutforth1/data-mf-ae/spheres_mu_2.50/SIGNED_DISTANCE_EXACT',
                     '/usr/workspace/cutforth1/data-mf-ae/spheres_mu_2.50/HEAVISIDE',
                     '/usr/workspace/cutforth1/data-mf-ae/spheres_mu_2.50/TANH_EPSILON0.0078125',
                     '/usr/workspace/cutforth1/data-mf-ae/spheres_mu_2.50/TANH_EPSILON0.015625',
                     '/usr/workspace/cutforth1/data-mf-ae/spheres_mu_2.50/TANH_EPSILON0.03125',
                     '/usr/workspace/cutforth1/data-mf-ae/spheres_mu_2.50/TANH_EPSILON0.0625',
                     '/usr/workspace/cutforth1/data-mf-ae/spheres_mu_2.50/TANH_EPSILON0.125',
                     '/usr/workspace/cutforth1/data-mf-ae/spheres_mu_2.50/TANH_EPSILON0.25',
                     ],
        'seed': [4],
        'max-train-samples': [50, 100, 200, 400, 800, 1600, 2000],
    }

    const_args = {
        'model-type': 'baseline',
        'dim': 32,
        'dataset-type': 'volumetric',
        'num-dl-workers': 0,
        'batch-size': 1,
        'num-epochs': 100,
        'lr': 1e-4,
        'loss': 'l1',
        'dim-mults': '1 2 4 8 8 8',
        'block-type': 1,
        'z-channels': 4,
        'max-samples': 25_000,
    }

    # Generate the Cartesian product of the values in search_args
    keys, values = zip(*search_args.items())
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]

    # Iterate over the combinations
    for i, combination in enumerate(combinations):
        run_name = create_param_str(i, combination)
        args_dict = {**const_args, **combination}

        create_run_script(i, run_name, args_dict)

        if i == 0:
            create_debug_script(args_dict)

    create_run_all_script(len(combinations))


def create_param_str(i, combination):
    param_str = "_".join([f"{key}{str(value)}" for key, value in combination.items()])
    param_str = param_str.replace("/usr/workspace/cutforth1/data-mf-ae/patched_hit_experiment/", "")
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
        f.write("source /usr/workspace/cutforth1/anaconda/bin/activate\n")
        f.write("export LD_LIBRARY_PATH=/opt/ibm/spectrumcomputing/lsf/10.1.0.10/linux3.10-glibc2.17-ppc64le-csm/lib\n")
        f.write("export PYTHONPATH=/usr/WS1/cutforth1/mf-ae\n")
        f.write("cd /usr/WS1/cutforth1/mf-ae\n")

        param_str = ""
        for k, v in args_dict.items():
            param_str += f" --{k} {v}"

        f.write(f"conda run -n genmodel_env accelerate launch ./src/main/main_train.py {param_str}\n")

    run_time = 720

    # Create corresponding .bsub
    with open(f"run_training_v{VERSION}_{i}.bsub", "w") as f:
        f.write(f"#BSUB -W {run_time}\n")
        f.write("#BSUB -G stanford\n")
        f.write("#BSUB -q pbatch\n")
        f.write(f"jsrun -n 1 -r 1 -a 1 -c 40 -g 4 ./run_training_v{VERSION}_{i}.sh\n")
        f.write(f'echo "Job complete at $(date)"\n')

    os.chmod(f"run_training_v{VERSION}_{i}.sh", 0o755)

    print(f"Created run script for combination {i} with run time {run_time}")


def create_debug_script(args_dict):
    run_name = f"run_debug_v{VERSION}"
    args_dict['run-name'] = run_name

    param_str = ""
    for k, v in args_dict.items():
        param_str += f" --{k} {v}"

    # Create .sh
    with open(f"run_training_v{VERSION}_debug.sh", "w") as f:
        f.write("source /usr/workspace/cutforth1/anaconda/bin/activate\n")
        f.write("export LD_LIBRARY_PATH=/opt/ibm/spectrumcomputing/lsf/10.1.0.10/linux3.10-glibc2.17-ppc64le-csm/lib\n")
        f.write("export PYTHONPATH=/usr/WS1/cutforth1/mf-ae\n")
        f.write("cd /usr/WS1/cutforth1/mf-ae\n")
        f.write(f"conda run -n genmodel_env accelerate launch ./src/main/main_train.py --debug {param_str}\n")

    # Create corresponding .bsub
    with open(f"run_training_v{VERSION}_debug.bsub", "w") as f:
        f.write("#BSUB -W 30\n")
        f.write("#BSUB -G stanford\n")
        f.write("#BSUB -q pdebug\n")
        f.write(f"jsrun -n 1 -r 1 -a 1 -c 40 -g 4 ./run_training_v{VERSION}_debug.sh\n")
        f.write(f'echo "Job complete at $(date)"\n')

    # Permission of .sh
    os.chmod(f"run_training_v{VERSION}_debug.sh", 0o755)

    print(f"Created debug run script")


def create_run_all_script(num_runs: int):
    with open('run_all.sh', 'w') as f:
        f.write('#!/bin/bash\n')
        for i in range(num_runs):
            f.write(f'bsub run_training_v{VERSION}_{i}.bsub\n')
        f.write('echo "All jobs submitted"')

    os.chmod("run_all.sh", 0o755)


if __name__ == '__main__':
    main()
