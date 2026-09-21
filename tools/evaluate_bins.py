import argparse
import csv
import math
import os
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


CLASS_TO_ID = {
    "Car": 0,
    "Pedestrian": 1,
    "Cyclist": 2,
}


def parse_boundaries(value):
    raw_boundaries = [item.strip() for item in value.split(",")]
    if len(raw_boundaries) < 2:
        raise ValueError("At least two bin boundaries are required")

    boundaries = []
    for index, item in enumerate(raw_boundaries):
        if item.lower() in ("inf", "+inf", "infinity"):
            if index != len(raw_boundaries) - 1:
                raise ValueError("inf is only allowed as the final boundary")
            boundaries.append(None)
        else:
            number = float(item)
            if not math.isfinite(number) or number < 0:
                raise ValueError("Bin boundaries must be finite non-negative values")
            boundaries.append(number)

    finite_boundaries = [item for item in boundaries if item is not None]
    if any(
            right <= left
            for left, right in zip(finite_boundaries, finite_boundaries[1:])):
        raise ValueError("Bin boundaries must be strictly increasing")
    if boundaries[-1] is not None and len(boundaries) < 2:
        raise ValueError("At least one complete bin is required")
    return list(zip(boundaries[:-1], boundaries[1:]))


def _read_image_ids(split_file):
    with open(split_file, "r") as file_handle:
        return [int(line.strip()) for line in file_handle if line.strip()]


def _format_bound(value):
    return "inf" if value is None else "%g" % value


def _format_bin(range_type, lower, upper):
    symbol = "h2D" if range_type == "size" else "Z"
    unit = "px" if range_type == "size" else "m"
    if upper is None:
        return "%s >= %g %s" % (symbol, lower, unit)
    return "%g <= %s < %g %s" % (lower, symbol, upper, unit)


def _validate_paths(dataset_root, result_dir, split):
    label_dir = os.path.join(dataset_root, "training", "label_2")
    split_file = os.path.join(dataset_root, "ImageSets", split + ".txt")
    for path, description in (
            (label_dir, "KITTI label directory"),
            (split_file, "KITTI split file"),
            (result_dir, "prediction directory")):
        if not os.path.exists(path):
            raise FileNotFoundError("%s does not exist: %s" % (description, path))
    return label_dir, split_file


def evaluate(args):
    from lib.datasets.kitti.kitti_eval_python import kitti_common as kitti
    from lib.datasets.kitti.kitti_eval_python.eval import get_range_eval_result

    label_dir, split_file = _validate_paths(
        args.dataset_root, args.result_dir, args.split)
    image_ids = _read_image_ids(split_file)
    missing_predictions = [
        image_id for image_id in image_ids
        if not os.path.exists(
            os.path.join(args.result_dir, "%06d.txt" % image_id))
    ]
    if missing_predictions:
        raise FileNotFoundError(
            "%d prediction files are missing; first missing image id: %06d"
            % (len(missing_predictions), missing_predictions[0])
        )

    gt_annos = kitti.get_label_annos(label_dir, image_ids)
    dt_annos = kitti.get_label_annos(args.result_dir, image_ids)
    range_groups = (
        ("size", parse_boundaries(args.size_bins)),
        ("distance", parse_boundaries(args.distance_bins)),
    )
    rows = []

    for range_type, bins in range_groups:
        for lower, upper in bins:
            for class_name in args.classes:
                metrics = get_range_eval_result(
                    gt_annos,
                    dt_annos,
                    CLASS_TO_ID[class_name],
                    range_type,
                    lower,
                    upper,
                )
                rows.append({
                    "range_type": range_type,
                    "range": _format_bin(range_type, lower, upper),
                    "lower": _format_bound(lower),
                    "upper": _format_bound(upper),
                    "class": class_name,
                    "iou_threshold": metrics["iou_threshold"],
                    "num_gt": metrics["num_gt"],
                    "ap_r40_3d": metrics["3d_ap_r40"],
                    "max_recall_3d": metrics["3d_max_recall"],
                    "ap_r40_bev": metrics["bev_ap_r40"],
                    "max_recall_bev": metrics["bev_max_recall"],
                })

    output_csv = args.output_csv or os.path.join(
        args.result_dir, "bin_evaluation.csv")
    output_parent = os.path.dirname(os.path.abspath(output_csv))
    os.makedirs(output_parent, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("range_type | range | class | GT | AP_R40 3D | Rmax 3D | AP_R40 BEV | Rmax BEV")
    for row in rows:
        print(
            "{range_type} | {range} | {class} | {num_gt} | "
            "{ap_r40_3d:.4f} | {max_recall_3d:.2f} | "
            "{ap_r40_bev:.4f} | {max_recall_bev:.2f}".format(**row)
        )
    print("CSV saved to: %s" % output_csv)
    return rows


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate KITTI predictions by 2D-box height and GT depth bins")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="KITTI root containing training/label_2 and ImageSets")
    parser.add_argument(
        "--result-dir",
        required=True,
        help="Prediction directory containing one six-digit .txt file per image")
    parser.add_argument("--split", default="val", help="ImageSets split name")
    parser.add_argument(
        "--classes",
        nargs="+",
        choices=tuple(CLASS_TO_ID.keys()),
        default=list(CLASS_TO_ID.keys()))
    parser.add_argument(
        "--size-bins",
        default="0,25,40,inf",
        help="Comma-separated h2D boundaries in pixels")
    parser.add_argument(
        "--distance-bins",
        default="0,20,40,inf",
        help="Comma-separated GT Z boundaries in metres")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path; defaults to RESULT_DIR/bin_evaluation.csv")
    return parser


def main():
    args = build_parser().parse_args()
    try:
        evaluate(args)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit("Evaluation failed: %s" % error)


if __name__ == "__main__":
    main()
