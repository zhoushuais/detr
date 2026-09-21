import importlib.util
import unittest

import numpy as np


class BestCheckpointSelectionTests(unittest.TestCase):
    def test_car_metric_selects_car_moderate_value(self):
        from lib.datasets.kitti.evaluation_protocol import select_best_score

        scores = {"Car": 20.0, "Pedestrian": 30.0, "Cyclist": 40.0}

        self.assertEqual(
            select_best_score(scores, "car_moderate_3d_r40"),
            20.0,
        )

    def test_legacy_mean_metric_remains_available(self):
        from lib.datasets.kitti.evaluation_protocol import select_best_score

        scores = {"Car": 20.0, "Pedestrian": 30.0, "Cyclist": 40.0}

        self.assertEqual(
            select_best_score(scores, "mean3_moderate_3d_r40"),
            30.0,
        )

    def test_selection_rejects_missing_car_score(self):
        from lib.datasets.kitti.evaluation_protocol import select_best_score

        with self.assertRaisesRegex(ValueError, "Car"):
            select_best_score(
                {"Pedestrian": 30.0, "Cyclist": 40.0},
                "car_moderate_3d_r40",
            )


def make_annotation(names, bboxes, locations, truncated=None, occluded=None):
    count = len(names)
    return {
        "name": np.asarray(names),
        "bbox": np.asarray(bboxes, dtype=np.float64).reshape(-1, 4),
        "location": np.asarray(locations, dtype=np.float64).reshape(-1, 3),
        "truncated": np.asarray(
            truncated if truncated is not None else [0.0] * count,
            dtype=np.float64,
        ),
        "occluded": np.asarray(
            occluded if occluded is not None else [0] * count,
            dtype=np.int64,
        ),
    }


class RangeCleaningTests(unittest.TestCase):
    def test_size_bin_uses_gt_height_and_keeps_all_class_detections(self):
        from lib.datasets.kitti.evaluation_protocol import clean_data_by_range

        gt = make_annotation(
            ["Car", "Car", "Van", "DontCare"],
            [
                [0, 0, 20, 24.9],
                [0, 0, 20, 25.0],
                [0, 0, 20, 20.0],
                [0, 0, 30, 30.0],
            ],
            [[0, 0, 10], [0, 0, 10], [0, 0, 10], [0, 0, 0]],
        )
        detections = make_annotation(
            ["Car", "Car", "Pedestrian"],
            [
                [0, 0, 20, 10.0],
                [0, 0, 20, 100.0],
                [0, 0, 20, 10.0],
            ],
            [[0, 0, 10], [0, 0, 10], [0, 0, 10]],
        )

        num_valid, ignored_gt, ignored_dt, dontcare = clean_data_by_range(
            gt,
            detections,
            current_class=0,
            range_type="size",
            lower=0.0,
            upper=25.0,
        )

        self.assertEqual(num_valid, 1)
        self.assertEqual(ignored_gt, [0, 1, 1, -1])
        self.assertEqual(ignored_dt, [0, 0, -1])
        self.assertEqual(len(dontcare), 1)

    def test_distance_bin_uses_forward_z_not_euclidean_distance(self):
        from lib.datasets.kitti.evaluation_protocol import clean_data_by_range

        gt = make_annotation(
            ["Car"],
            [[0, 0, 20, 30]],
            [[100, 0, 10]],
        )
        detections = make_annotation([], [], [])

        num_valid, ignored_gt, _, _ = clean_data_by_range(
            gt,
            detections,
            current_class=0,
            range_type="distance",
            lower=0.0,
            upper=20.0,
        )

        self.assertEqual(num_valid, 1)
        self.assertEqual(ignored_gt, [0])

    def test_quality_filter_ignores_heavily_occluded_target(self):
        from lib.datasets.kitti.evaluation_protocol import clean_data_by_range

        gt = make_annotation(
            ["Cyclist"],
            [[0, 0, 20, 20]],
            [[0, 0, 10]],
            occluded=[3],
        )
        detections = make_annotation([], [], [])

        num_valid, ignored_gt, _, _ = clean_data_by_range(
            gt,
            detections,
            current_class=2,
            range_type="size",
            lower=0.0,
            upper=25.0,
        )

        self.assertEqual(num_valid, 0)
        self.assertEqual(ignored_gt, [1])

    def test_invalid_range_type_is_rejected(self):
        from lib.datasets.kitti.evaluation_protocol import clean_data_by_range

        empty = make_annotation([], [], [])

        with self.assertRaisesRegex(ValueError, "range_type"):
            clean_data_by_range(
                empty,
                empty,
                current_class=0,
                range_type="area",
                lower=0.0,
                upper=25.0,
            )

    @unittest.skipUnless(
        importlib.util.find_spec("numba"),
        "numba is required to execute the full KITTI evaluator",
    )
    def test_empty_range_returns_nan_metrics_without_crashing(self):
        from lib.datasets.kitti.kitti_eval_python.eval import get_range_eval_result

        empty = make_annotation([], [], [])

        result = get_range_eval_result(
            [empty],
            [empty],
            current_class=0,
            range_type="distance",
            lower=100.0,
            upper=None,
        )

        self.assertEqual(result["num_gt"], 0)
        self.assertTrue(np.isnan(result["3d_ap_r40"]))
        self.assertTrue(np.isnan(result["3d_max_recall"]))


class BinCommandTests(unittest.TestCase):
    def test_parse_boundaries_builds_half_open_bins(self):
        from tools.evaluate_bins import parse_boundaries

        self.assertEqual(
            parse_boundaries("0,25,40,inf"),
            [(0.0, 25.0), (25.0, 40.0), (40.0, None)],
        )

    def test_parse_boundaries_rejects_non_increasing_values(self):
        from tools.evaluate_bins import parse_boundaries

        with self.assertRaisesRegex(ValueError, "increasing"):
            parse_boundaries("0,40,25,inf")


class ModelProfileTests(unittest.TestCase):
    def test_profile_parser_defaults_to_cuda(self):
        from tools.profile_model import build_parser

        args = build_parser().parse_args(["--config", "model.yaml"])

        self.assertEqual(args.device, "cuda")

    def test_count_parameters_separates_total_and_trainable(self):
        from tools.profile_model import count_parameters

        class FakeParameter:
            def __init__(self, count, requires_grad):
                self.count = count
                self.requires_grad = requires_grad

            def numel(self):
                return self.count

        class FakeModel:
            def parameters(self):
                return [FakeParameter(10, True), FakeParameter(5, False)]

        self.assertEqual(count_parameters(FakeModel()), (15, 10))

    def test_convert_macs_uses_two_flops_per_mac(self):
        from tools.profile_model import convert_macs

        self.assertEqual(convert_macs(3_000_000_000), (3.0, 6.0))


if __name__ == "__main__":
    unittest.main()
