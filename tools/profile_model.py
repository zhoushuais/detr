import argparse
import os
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def count_parameters(model):
    parameters = list(model.parameters())
    total = sum(parameter.numel() for parameter in parameters)
    trainable = sum(
        parameter.numel() for parameter in parameters
        if parameter.requires_grad
    )
    return total, trainable


def convert_macs(macs):
    gmacs = float(macs) / 1e9
    gflops = 2.0 * gmacs
    return gmacs, gflops


def build_parser():
    parser = argparse.ArgumentParser(
        description="Report MonoDETR parameters and inference complexity")
    parser.add_argument("--config", required=True, help="Model YAML config")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument(
        "--device",
        default="cuda",
        choices=("cuda", "cpu"),
        help="Profiling device; CUDA is required by the default custom ops")
    parser.add_argument(
        "--parameters-only",
        action="store_true",
        help="Skip MAC profiling; does not require thop")
    return parser


def profile(args):
    import torch
    import yaml

    from lib.helpers.model_helper import build_model

    with open(args.config, "r") as file_handle:
        cfg = yaml.load(file_handle, Loader=yaml.Loader)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run on the training server")
    device = torch.device(args.device)
    cfg["model"]["device"] = args.device
    model, _ = build_model(cfg["model"])
    model = model.to(device).eval()

    total, trainable = count_parameters(model)
    print("Total parameters:     %.3f M" % (total / 1e6))
    print("Trainable parameters: %.3f M" % (trainable / 1e6))

    if args.parameters_only:
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
        }

    try:
        from thop import profile as thop_profile
    except ImportError as error:
        raise RuntimeError(
            "THOP is required for MAC profiling. Install it with: pip install thop"
        ) from error

    images = torch.zeros(1, 3, args.height, args.width, device=device)
    calibs = torch.zeros(1, 3, 4, device=device)
    calibs[:, 0, 0] = 721.5377
    calibs[:, 1, 1] = 721.5377
    calibs[:, 2, 2] = 1.0
    img_sizes = torch.tensor(
        [[args.width, args.height]], dtype=torch.float32, device=device)
    macs, _ = thop_profile(
        model,
        inputs=(images, calibs, None, img_sizes, 0),
        verbose=False,
    )
    gmacs, gflops = convert_macs(macs)
    print("Input resolution:     %d x %d, batch size 1" % (
        args.width, args.height))
    print("GMACs:                %.3f" % gmacs)
    print("GFLOPs (2 FLOPs/MAC): %.3f" % gflops)
    print(
        "Note: THOP may not count CUDA/custom operators such as "
        "multi-scale deformable attention and deformable convolution. "
        "Use the same environment and tool version for every compared model."
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "gmacs": gmacs,
        "gflops": gflops,
    }


def main():
    args = build_parser().parse_args()
    try:
        profile(args)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit("Profiling failed: %s" % error)


if __name__ == "__main__":
    main()
