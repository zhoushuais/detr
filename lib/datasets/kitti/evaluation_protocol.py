import math


CLASS_NAMES = ("car", "pedestrian", "cyclist", "van", "person_sitting", "truck")
ALIASES = {
    "car": "van",
    "pedestrian": "person_sitting",
}
BEST_SELECTION_METRICS = (
    "car_moderate_3d_r40",
    "mean3_moderate_3d_r40",
)


def select_best_score(moderate_3d_r40, metric):
    if metric not in BEST_SELECTION_METRICS:
        raise ValueError(
            "Unsupported best-selection metric %r. Expected one of: %s"
            % (metric, ", ".join(BEST_SELECTION_METRICS))
        )

    if metric == "car_moderate_3d_r40":
        if "Car" not in moderate_3d_r40:
            raise ValueError(
                "Car Moderate AP_R40 3D is required for best checkpoint selection."
            )
        return float(moderate_3d_r40["Car"])

    required_classes = ("Car", "Pedestrian", "Cyclist")
    missing_classes = [
        class_name for class_name in required_classes
        if class_name not in moderate_3d_r40
    ]
    if missing_classes:
        raise ValueError(
            "Three-class Moderate AP_R40 3D selection is missing: %s"
            % ", ".join(missing_classes)
        )
    return sum(float(moderate_3d_r40[name]) for name in required_classes) / 3.0


def _value_in_range(value, lower, upper):
    return value >= lower and (upper is None or value < upper)


def _target_value(gt_anno, index, range_type):
    if range_type == "size":
        bbox = gt_anno["bbox"][index]
        return float(bbox[3] - bbox[1])
    if range_type == "distance":
        return float(gt_anno["location"][index][2])
    raise ValueError("range_type must be 'size' or 'distance', got %r" % range_type)


def _validate_range(range_type, lower, upper):
    if range_type not in ("size", "distance"):
        raise ValueError("range_type must be 'size' or 'distance', got %r" % range_type)
    if not math.isfinite(lower) or lower < 0:
        raise ValueError("lower must be a finite non-negative number")
    if upper is not None and (not math.isfinite(upper) or upper <= lower):
        raise ValueError("upper must be greater than lower or None")


def clean_data_by_range(
        gt_anno,
        dt_anno,
        current_class,
        range_type,
        lower,
        upper,
        max_occlusion=2,
        max_truncation=0.5):
    _validate_range(range_type, lower, upper)
    if current_class < 0 or current_class > 2:
        raise ValueError("current_class must be 0 (Car), 1 (Pedestrian), or 2 (Cyclist)")

    current_name = CLASS_NAMES[current_class]
    alias_name = ALIASES.get(current_name)
    ignored_gt = []
    ignored_dt = []
    dontcare_bboxes = []
    num_valid_gt = 0

    for index, raw_name in enumerate(gt_anno["name"]):
        gt_name = raw_name.lower()
        is_current_class = gt_name == current_name
        is_alias = alias_name is not None and gt_name == alias_name
        quality_ok = (
            gt_anno["occluded"][index] <= max_occlusion
            and gt_anno["truncated"][index] <= max_truncation
        )
        in_range = _value_in_range(
            _target_value(gt_anno, index, range_type),
            lower,
            upper,
        )

        if is_current_class and quality_ok and in_range:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif is_alias or is_current_class:
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)

        if gt_name == "dontcare":
            dontcare_bboxes.append(gt_anno["bbox"][index])

    for raw_name in dt_anno["name"]:
        ignored_dt.append(0 if raw_name.lower() == current_name else -1)

    return num_valid_gt, ignored_gt, ignored_dt, dontcare_bboxes


def make_range_cleaner(range_type, lower, upper):
    _validate_range(range_type, lower, upper)

    def cleaner(gt_anno, dt_anno, current_class, difficulty):
        del difficulty
        return clean_data_by_range(
            gt_anno,
            dt_anno,
            current_class,
            range_type,
            lower,
            upper,
        )

    return cleaner
