from pathlib import Path


def project_dir():
    return Path(__file__).resolve().parent.parent


def output_dir_base():
    outdir = project_dir() / 'output'
    outdir.mkdir(exist_ok=True)
    return outdir


def local_data_dir():
    return project_dir() / 'data'