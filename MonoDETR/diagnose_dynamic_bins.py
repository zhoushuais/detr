"""
[P2-DBDU] Diagnostic: did the Dynamic Bins actually adapt, or stay frozen at LID?

Run from the MonoDETR/ source root (same place you run tools/train_val.py):

    python diagnose_dynamic_bins.py --config configs/monodetr.yaml \
        --ckpt outputs/monodetr/checkpoint_best.pth

Two tiers:
  Tier 1 (always, no GPU/data needed): inspect bin_predictor weights in the
          checkpoint. Decisive question -> is the LAST linear weight still ~0?
            * ~0  -> bins are input-INDEPENDENT (same partition for every image).
                     If bias also ~= LID init -> bins FROZEN at LID, the dynamic
                     mechanism never activated (gradient too weak to move it).
                     If bias moved -> degenerated to a different FIXED partition.
            * >0  -> bins respond to image content. Tier 2 quantifies how much.
  Tier 2 (needs GPU + data): run a few val images, capture per-image bin_edges
          via a forward hook on depth_predictor, report cross-image spread and
          drift from the LID reference partition.
"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import argparse

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Find the project root that actually contains lib/ (works whether this file
# sits in the source root or in tools/, and regardless of CWD).
for _cand in (os.getcwd(), BASE_DIR, os.path.dirname(BASE_DIR)):
    if os.path.isdir(os.path.join(_cand, "lib")) and _cand not in sys.path:
        sys.path.insert(0, _cand)


def safe_load(path):
    # torch>=2.6 flipped weights_only default to True; older torch lacks the kwarg.
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def lid_reference_edges(depth_min, depth_max, num_bins):
    # Baseline LID bin edges (also exactly what the zero-init dynamic head
    # reproduces at iter 0). edge[k] for k=0..num_bins, edge[0]=dmin, edge[-1]=dmax.
    k = torch.arange(0, num_bins + 1, dtype=torch.float64)
    bin_size = 2.0 * (depth_max - depth_min) / (num_bins * (1.0 + num_bins))
    edges = depth_min + bin_size * (((2 * k + 1) ** 2) - 1.0) / 8.0
    return edges  # (num_bins+1,)


def find_keys(state, needle):
    return {k: v for k, v in state.items() if needle in k}


# ----------------------------------------------------------------------------
# Tier 1: weight inspection
# ----------------------------------------------------------------------------
def tier1(ckpt_path, depth_min, depth_max):
    print("=" * 72)
    print("TIER 1  -  bin_predictor weight inspection")
    print("=" * 72)
    ckpt = safe_load(ckpt_path)
    state = ckpt.get("model_state", ckpt)
    print("checkpoint epoch       :", ckpt.get("epoch", "?"))
    print("checkpoint best_result :", ckpt.get("best_result", "?"),
          "@ epoch", ckpt.get("best_epoch", "?"))

    bp = find_keys(state, "bin_predictor")
    if not bp:
        print("\n[!] No 'bin_predictor.*' keys found in this checkpoint.")
        print("    -> This checkpoint was trained with use_dynamic_bins=False,")
        print("       i.e. it is a BASELINE run, not a Dynamic Bins run.")
        print("    Re-check which checkpoint / config produced it.")
        return None
    print("\nfound bin_predictor params:")
    for k, v in bp.items():
        print("   {:50s} {}".format(k, tuple(v.shape)))

    # last linear = highest layer index in the Sequential (Linear, ReLU, Linear)
    w_keys = sorted(k for k in bp if k.endswith(".weight"))
    b_keys = sorted(k for k in bp if k.endswith(".bias"))
    last_w_key = w_keys[-1]
    last_b_key = b_keys[-1]
    last_w = bp[last_w_key].double()
    last_b = bp[last_b_key].double()
    num_bins = last_w.shape[0]

    w_fro = last_w.norm().item()
    w_absmax = last_w.abs().max().item()
    w_absmean = last_w.abs().mean().item()

    # init bias = log(normalized LID widths), widths ∝ (i+1)
    lid_widths = torch.arange(1, num_bins + 1, dtype=torch.float64)
    init_bias = torch.log(lid_widths / lid_widths.sum())
    bias_drift = (last_b - init_bias).abs()

    print("\n-- last linear (the image->bins map) --   key:", last_w_key)
    print("   weight  Frobenius norm : {:.6f}   (init = 0.000000)".format(w_fro))
    print("   weight  max |w|        : {:.6f}".format(w_absmax))
    print("   weight  mean |w|       : {:.6f}".format(w_absmean))
    print("   bias    drift vs LID   : mean {:.4f}  max {:.4f}  (init drift = 0)"
          .format(bias_drift.mean().item(), bias_drift.max().item()))

    # verdict
    print("\n-- VERDICT --")
    WEIGHT_DEAD = 1e-3   # below this the image->bins map is effectively off
    BIAS_STILL = 0.05    # below this the global partition barely moved from LID
    if w_fro < WEIGHT_DEAD:
        print("   Last-linear weight is ~0  =>  bins are INPUT-INDEPENDENT")
        print("   (every image gets the SAME partition; not 'dynamic' per image).")
        if bias_drift.mean().item() < BIAS_STILL:
            print("   Bias also ~= LID init  =>  bins FROZEN at LID.")
            print("   The dynamic mechanism never activated. Likely cause: the")
            print("   gradient reaching bin_predictor (only via weighted_depth ->")
            print("   pos-encoding + aux depth-map loss) is too weak to move a")
            print("   zero-initialised head. This is an OPTIMIZATION problem.")
        else:
            print("   Bias moved  =>  degenerated to a DIFFERENT fixed partition")
            print("   (global re-tuning of LID, still identical across images).")
        print("   => 'Dynamic' is effectively NOT happening. Tier 2 will show")
        print("      ~0 cross-image spread.")
    else:
        print("   Last-linear weight is non-zero  =>  bins DO respond to image")
        print("   content (genuinely per-image). Run Tier 2 to quantify the")
        print("   per-image spread and the drift from LID.")
    return num_bins


# ----------------------------------------------------------------------------
# Tier 2: run a few real images, capture per-image bin_edges
# ----------------------------------------------------------------------------
def tier2(cfg_path, ckpt_path, depth_min, depth_max, n_show=6):
    print("\n" + "=" * 72)
    print("TIER 2  -  per-image bin_edges on real val images")
    print("=" * 72)
    import yaml
    from lib.helpers.model_helper import build_model
    from lib.helpers.dataloader_helper import build_dataloader
    from lib.helpers.save_helper import load_checkpoint

    class _L:  # minimal logger shim for load_checkpoint
        def info(self, *a, **k):
            print(*a)

    cfg = yaml.load(open(cfg_path, "r"), Loader=yaml.Loader)
    if not cfg["model"].get("use_dynamic_bins", False):
        print("[!] config has use_dynamic_bins=False -> model has no bin_predictor.")
        print("    Set it True (matching the trained ckpt) to run Tier 2.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = build_dataloader(cfg["dataset"])
    model, _ = build_model(cfg["model"])
    load_checkpoint(model=model, optimizer=None, filename=ckpt_path,
                    map_location=device, logger=_L())
    model.to(device).eval()

    captured = {}

    def hook(_m, _inp, out):
        captured["bin_edges"] = out[-1]  # forward returns (...; bin_edges) last

    h = model.depth_predictor.register_forward_hook(hook)

    with torch.no_grad():
        inputs, calibs, targets, info = next(iter(test_loader))
        inputs = inputs.to(device)
        calibs = calibs.to(device)
        img_sizes = info["img_size"].to(device)
        model(inputs, calibs, targets, img_sizes, dn_args=0)
    h.remove()

    edges = captured.get("bin_edges")
    if edges is None:
        print("[!] hook captured None -> use_dynamic_bins is off in the model.")
        return
    edges = edges.double().cpu()                 # (B, num_bins+1)
    B, E = edges.shape
    num_bins = E - 1
    lid = lid_reference_edges(depth_min, depth_max, num_bins)  # (num_bins+1,)

    # probe a handful of boundary indices spread across the range
    probe = [i for i in (5, 10, 20, 40, 60, num_bins) if i <= num_bins]
    print("\nBoundary depth (m) at selected bin indices, per image:")
    header = "  img |" + "".join("  idx{:>3d}".format(p) for p in probe)
    print(header)
    print("  LID  |" + "".join("  {:6.2f}".format(lid[p].item()) for p in probe)
          + "   <- reference / iter-0 init")
    for b in range(min(n_show, B)):
        row = "".join("  {:6.2f}".format(edges[b, p].item()) for p in probe)
        print("  {:>3d}  |{}".format(b, row))

    # cross-image spread per boundary, and drift from LID
    std_per_edge = edges.std(dim=0)              # (num_bins+1,)
    drift = (edges - lid.unsqueeze(0)).abs()     # (B, num_bins+1)
    print("\n-- cross-image spread (std over the {} images in this batch) --".format(B))
    print("   max  std over boundaries : {:.4f} m".format(std_per_edge.max().item()))
    print("   mean std over boundaries : {:.4f} m".format(std_per_edge.mean().item()))
    print("-- drift from LID reference --")
    print("   mean |edge - LID|        : {:.4f} m".format(drift.mean().item()))
    print("   max  |edge - LID|        : {:.4f} m".format(drift.max().item()))

    print("\n-- VERDICT --")
    if std_per_edge.max().item() < 0.05:
        print("   Spread ~0  =>  all images share ONE partition (NOT per-image).")
        if drift.mean().item() < 0.05:
            print("   And it equals LID  =>  bins FROZEN at LID init. Dynamic OFF.")
        else:
            print("   But it drifted from LID  =>  one global re-tuned partition.")
    else:
        print("   Non-trivial spread  =>  bins ARE genuinely per-image.")
        print("   If metrics still didn't improve, the lever is weak (bins only")
        print("   feed pos-encoding + aux loss, not the final depth_ave) -> move")
        print("   to the next increment that touches depth_ave directly.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/monodetr.yaml")
    ap.add_argument("--ckpt", default="outputs/monodetr/checkpoint_best.pth")
    ap.add_argument("--depth_min", type=float, default=1e-3)
    ap.add_argument("--depth_max", type=float, default=60.0)
    ap.add_argument("--skip_tier2", action="store_true",
                    help="only inspect weights, do not build model / load data")
    args = ap.parse_args()

    assert os.path.exists(args.ckpt), "checkpoint not found: " + args.ckpt
    found = tier1(args.ckpt, args.depth_min, args.depth_max)

    if args.skip_tier2:
        return
    if found is None:
        print("\n(Skipping Tier 2: no bin_predictor in checkpoint.)")
        return
    try:
        tier2(args.config, args.ckpt, args.depth_min, args.depth_max)
    except Exception as e:
        import traceback
        print("\n[Tier 2 failed -- Tier 1 result above still stands]")
        print("reason:", repr(e))
        traceback.print_exc()


if __name__ == "__main__":
    main()
