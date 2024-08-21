import unittest
import numpy as np
from src.evaluation.eval_ae_error import hausdorff_distance

class TestHausdorffDistance(unittest.TestCase):

    def test_identical_grids(self):
        gt_patch = np.ones((5, 5, 5))
        pred_patch = np.ones((5, 5, 5))
        self.assertEqual(hausdorff_distance(gt_patch, pred_patch), 0)

    def test_single_point(self):
        gt_patch = np.zeros((5, 5, 5))
        pred_patch = np.zeros((5, 5, 5))
        gt_patch[2, 2, 2] = 1
        pred_patch[2, 2, 2] = 1
        self.assertEqual(hausdorff_distance(gt_patch, pred_patch), 0)

    def test_different_points(self):
        gt_patch = np.zeros((5, 5, 5))
        pred_patch = np.zeros((5, 5, 5))
        gt_patch[0, 0, 0] = 1
        pred_patch[4, 4, 4] = 1
        self.assertEqual(hausdorff_distance(gt_patch, pred_patch), np.sqrt(3 * (4**2)))

    def test_empty_and_non_empty(self):
        gt_patch = np.zeros((5, 5, 5))
        pred_patch = np.zeros((5, 5, 5))
        pred_patch[2, 2, 2] = 1
        self.assertIs(hausdorff_distance(gt_patch, pred_patch), np.nan)

    def test_complex_case(self):
        gt_patch = np.zeros((5, 5, 5))
        pred_patch = np.zeros((5, 5, 5))
        gt_patch[1, 1, 1] = 1
        gt_patch[3, 3, 3] = 1
        pred_patch[1, 1, 1] = 1
        pred_patch[4, 4, 4] = 1
        self.assertEqual(hausdorff_distance(gt_patch, pred_patch), np.sqrt(3 * (1**2)))

if __name__ == '__main__':
    unittest.main()