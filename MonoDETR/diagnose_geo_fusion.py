"""
[P2-DBDU] Diagnostic: did the geometric-depth fusion GATE actually open?

Gate is now CLAMP-LINEAR: alpha = clamp(geo_fusion_gate, 0, 1), init 0.05, with a
non-vanishing gradient (the old sigmoid(-4) init was gradient-frozen and never
moved). loss-sigma is decoupled, so the gate opening only refines the point
estimate. After a use_geo_depth_fusion=True run, read alpha to interpret:
  * alpha climbed clearly above init (e.g. 0.2+) -> fusion ENGAGED. If AP improved,
    geometry sigma-weighting helps; if AP flat, it engaged but is neutral.
  * alpha still ~= init 0.05 -> with a FREE gradient this means opening the gate
    did NOT reduce loss => geometry sigma-weighting genuinely doesn't help. Clean
    negative (a real test this time, unlike the frozen-sigmoid run). Close it.

Run from the MonoDETR/ source root (no GPU / no data needed):

    python diagnose_geo_fusion.py --ckpt outputs/monodetr/checkpoint_best.pth
"""
import warnings
warnings.filterwarnings("ignore")

import os
import math
import argparse

import torch

GATE_INIT = 0.05  # configs/monodetr.yaml: geo_fusion_gate_init default (clamp-linear)


def safe_load(path):
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def find_scalar(state, needle):
    for k, v in state.items():
        if needle in k:
            return k, v
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/monodetr/checkpoint_best.pth")
    args = ap.parse_args()
    assert os.path.exists(args.ckpt), "checkpoint not found: " + args.ckpt

    ckpt = safe_load(args.ckpt)
    state = ckpt.get("model_state", ckpt)
    print("=" * 64)
    print("P2-DBDU geometric-depth fusion -- gate inspection")
    print("=" * 64)
    print("checkpoint epoch       :", ckpt.get("epoch", "?"))
    print("checkpoint best_result :", ckpt.get("best_result", "?"),
          "@ epoch", ckpt.get("best_epoch", "?"))

    gk, gate = find_scalar(state, "geo_fusion_gate")
    sk, lsh = find_scalar(state, "geo_log_sigma_h")
    mk, lvm = find_scalar(state, "depthmap_log_var")

    if gate is None:
        print("\n[!] No 'geo_fusion_gate' in this checkpoint.")
        print("    -> trained with use_geo_depth_fusion=False (baseline), or an")
        print("       older code version. Re-check the run / config.")
        return

    gate = float(gate)
    alpha = min(1.0, max(0.0, gate))   # clamp-linear gate
    print("\nlearned scalars:")
    print("   geo_fusion_gate   : {:+.4f}   (init {:+.2f})   -> alpha = clamp(0,1) = {:.4f}"
          .format(gate, GATE_INIT, alpha))
    if lsh is not None:
        print("   geo_log_sigma_h   : {:+.4f}   -> sigma_h = {:.3f} px".format(float(lsh), math.exp(float(lsh))))
    if lvm is not None:
        print("   depthmap_log_var  : {:+.4f}".format(float(lvm)))

    moved = gate - GATE_INIT
    print("\n-- VERDICT --")
    print("   gate moved {:+.4f} from init (0.05).   alpha = {:.4f}".format(moved, alpha))
    if alpha >= 0.20:
        print("   Gate OPENED (alpha {:.2f}) => fusion genuinely engaged this time".format(alpha))
        print("   (free gradient, unlike the frozen-sigmoid run).")
        print("   -> If AP improved (esp. Ped/Cyc Mod/Hard): geometry sigma-")
        print("      weighting HELPS, keep it and tune.")
        print("   -> If AP ~= baseline: engaged but NEUTRAL. Clean negative, close")
        print("      the increment and move to the next P2 lever (ADPM residual).")
    elif moved <= -0.03:
        print("   Gate CLOSED toward 0 (alpha {:.2f}) => the optimizer actively".format(alpha))
        print("   pushed geometry OUT. Geometry fusion HURTS the point estimate.")
        print("   Clean negative -> close the increment.")
    else:
        print("   Gate ~= init (alpha {:.2f}). With a FREE (non-saturated) gradient,".format(alpha))
        print("   staying put means opening it did not reduce loss => geometry")
        print("   sigma-weighting is NEUTRAL. This IS a real test now. Clean")
        print("   negative -> close the increment, move to the next P2 lever.")


if __name__ == "__main__":
    main()
