import logging

from matplotlib import pyplot as plt
from optuna.visualization.matplotlib import plot_contour, plot_edf, plot_intermediate_values, plot_optimization_history, \
    plot_parallel_coordinate, plot_param_importances, plot_rank, plot_slice, plot_timeline

logger = logging.getLogger(__name__)


def finalise_study(study, outdir):
    logger.info("Study statistics: ")
    logger.info(f"  Number of finished trials: {len(study.trials)}")
    logger.info("Best trial:")
    trial = study.best_trial
    logger.info("  Value: ", trial.value)
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info("    {}: {}".format(key, value))
    plot_tuning_results(study, outdir)


def plot_tuning_results(study, outdir):
    logger.info("Plotting output:")

    plot_contour(study)
    fig = plt.gcf()
    fig.set_size_inches(12, 12)
    fig.savefig(outdir / 'contour.png', dpi=200)
    plt.close()

    plot_edf(study)
    plt.gcf().savefig(outdir / 'edf.png')
    plt.close()

    plot_intermediate_values(study)
    plt.gcf().savefig(outdir / 'intermediate_values.png')
    plt.close()

    plot_optimization_history(study)
    plt.gcf().savefig(outdir / 'optimization_history.png')
    plt.close()

    plot_parallel_coordinate(study)
    plt.gcf().savefig(outdir / 'parallel_coordinate.png')
    plt.close()

    plot_param_importances(study)
    plt.gcf().savefig(outdir / 'param_importances.png')
    plt.close()

    plot_rank(study)
    fig = plt.gcf()
    fig.set_size_inches(12, 12)
    fig.savefig(outdir / 'rank.png', dpi=200)
    plt.close()

    plot_slice(study)
    plt.gcf().savefig(outdir / 'slice.png', dpi=200)
    plt.close()

    plot_timeline(study)
    plt.gcf().savefig(outdir / 'timeline.png')
    plt.close()
