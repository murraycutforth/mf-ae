import unittest
import tempfile

from src.plotting_utils import write_isosurface_plot, write_slice_plot
from src.paths import output_dir_base, local_data_dir


class TestPlotAllData(unittest.TestCase):

    def test_plot_all_data(self):
        # Get list of all data
        data_dir = local_data_dir()
        filenames = list(data_dir.glob("*.npy"))

        output_dir_surfaces = output_dir_base() / 'isosurface_plots'
        output_dir_surfaces.mkdir(exist_ok=True)

        output_dir_slices = output_dir_base() / 'slice_plots'
        output_dir_slices.mkdir(exist_ok=True)

        for filename in filenames:
            write_slice_plot(restart_path=filename,
                             dx=256,
                                  outdir=output_dir_slices,
                                  verbose=True)

            write_isosurface_plot(restart_path=filename,
                             dx=256,
                             outdir=output_dir_surfaces,
                             verbose=True)

