import itertools
import os

VERSION = 5

# Dataset: ellipse
# Search space: dim, interface-representation
# dim: [4, 8, 16]
# Interface representation: [heaviside, diffuse, sdf]


def main():
    search_args = {
        'dim': [4, 8, 16],
        'interface-representation': ['heaviside', 'diffuse', 'sdf'],
    }

    const_args = {
        'dataset-type': 'ellipse',
        'num-dl-workers': 0,
        'batch-size': 1,
        'num-epochs': 10,
        'save-and-sample-every': 1,
        'lr': 1e-4,
        'loss': 'l1',
        'dim-mults': '1 2 4 8 8',
        'block-type': 1,
        'z-channels': 4,
    }

    # Generate the Cartesian product of the values in search_args
    keys, values = zip(*search_args.items())
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]

    # Iterate over the combinations
    for i, combination in enumerate(combinations):
        run_name = create_param_str(i, combination)
        args_dict = {**const_args, **combination}

        # Number of steps should be constant, so number of epochs should be inversely proportional to the proportion of training data
        #args_dict['num-epochs'] = int(40 / args_dict['train-data-proportion'])
        #args_dict['save-every'] = int(args_dict['num-epochs'] / 5)

        create_run_script(i, run_name, args_dict)

        if i == 0:
            create_debug_script(args_dict)

    create_run_all_script(len(combinations))


def create_param_str(i, combination):
    param_str = "_".join([f"{key}{str(value)}" for key, value in combination.items()])
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
    with open(f"run_ae_training_v{VERSION}_{i}.sh", "w") as f:
        f.write("source /usr/workspace/cutforth1/anaconda/bin/activate\n")
        f.write("export LD_LIBRARY_PATH=/opt/ibm/spectrumcomputing/lsf/10.1.0.10/linux3.10-glibc2.17-ppc64le-csm/lib\n")
        f.write("export PYTHONPATH=/g/g91/cutforth1/mf-ae\n")
        f.write("cd /g/g91/cutforth1/mf-ae\n")

        param_str = ""
        for k, v in args_dict.items():
            param_str += f" --{k} {v}"

        f.write(f"conda run -n genmodel_env accelerate launch ./src/main/main_train.py {param_str}\n")

    run_time = 720

    # Create corresponding .bsub
    with open(f"run_ae_training_v{VERSION}_{i}.bsub", "w") as f:
        f.write(f"#BSUB -W {run_time}\n")
        f.write("#BSUB -G stanford\n")
        f.write("#BSUB -q pbatch\n")
        f.write(f"jsrun -n 1 -r 1 -a 1 -c 40 -g 4 ./run_ae_training_v{VERSION}_{i}.sh\n")
        f.write(f'echo "Job complete at $(date)"\n')

    os.chmod(f"run_ae_training_v{VERSION}_{i}.sh", 0o755)

    print(f"Created run script for combination {i} with run time {run_time}")


def create_debug_script(args_dict):
    run_name = f"run_debug_v{VERSION}"
    args_dict['run-name'] = run_name

    param_str = ""
    for k, v in args_dict.items():
        param_str += f" --{k} {v}"

    # Create .sh
    with open(f"run_ae_training_v{VERSION}_debug.sh", "w") as f:
        f.write("source /usr/workspace/cutforth1/anaconda/bin/activate\n")
        f.write("export LD_LIBRARY_PATH=/opt/ibm/spectrumcomputing/lsf/10.1.0.10/linux3.10-glibc2.17-ppc64le-csm/lib\n")
        f.write("export PYTHONPATH=/g/g91/cutforth1/mf-ae\n")
        f.write("cd /g/g91/cutforth1/mf-ae\n")
        f.write(f"conda run -n genmodel_env accelerate launch ./src/main/main_train.py --debug {param_str}\n")

    # Create corresponding .bsub
    with open(f"run_ae_training_v{VERSION}_debug.bsub", "w") as f:
        f.write("#BSUB -W 30\n")
        f.write("#BSUB -G stanford\n")
        f.write("#BSUB -q pdebug\n")
        f.write(f"jsrun -n 1 -r 1 -a 1 -c 40 -g 4 ./run_ae_training_v{VERSION}_debug.sh\n")
        f.write(f'echo "Job complete at $(date)"\n')

    # Permission of .sh
    os.chmod(f"run_ae_training_v{VERSION}_debug.sh", 0o755)

    print(f"Created debug run script")


def create_run_all_script(num_runs: int):
    with open('run_all.sh', 'w') as f:
        f.write('#!/bin/bash\n')
        for i in range(num_runs):
            f.write(f'bsub run_ae_training_v{VERSION}_{i}.bsub\n')
        f.write('echo "All jobs submitted"')

    os.chmod("run_all.sh", 0o755)


if __name__ == '__main__':
    main()
