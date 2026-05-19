# ------------------------------------------------------------------------
# Improvement Direction 1 (P1): SCA-FPN
# Spatial-Channel Attention Feature Pyramid Network for monocular 3D detection
#
# Motivation:
#   MonoDETR feeds 4-level multi-scale features {1/8, 1/16, 1/32, 1/64} into the
#   Deformable-Attention encoder, but the levels are NOT semantically aligned —
#   they are just projected by 1x1 conv and concatenated. Small / far-away
#   objects (pedestrians, cyclists) lose detail at 1/32 and are barely visible
#   at 1/64, while shallow 1/8 level has rich geometry but weak semantics.
#
# Idea (after YANG SHUAIBING et al. SCA-FPN, 2026; LI CHENGSHUO 2024 multi-scale):
#   1. Per-level Channel Attention (CBAM-style) suppresses redundant channels.
#   2. A Spatial Attention computed on the HIGHEST level is broadcast to lower
#      levels, providing top-down semantic guidance that low layers couldn't
#      learn on their own.
#   3. Each lower level fuses (its own local SA + upsampled high-level SA)
#      through sigmoid gating, then is residually added back to preserve
#      pre-trained statistics (important for stability when finetuning).
#
# Plug-in point: between `input_proj` and the downstream modules in
# `monodetr.py forward()`. Both depth_predictor and depthaware_transformer
# operate on the SAME `srcs` list, so they automatically benefit.
#
# Toggle: controlled by config flag `use_sca_fpn` (default False -> identity).
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """CBAM-style channel attention: global avg + max pool -> shared MLP -> sigmoid."""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        hidden = max(in_channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        attn = torch.sigmoid(self.mlp(avg) + self.mlp(mx))   # (B, C, 1, 1)
        return attn


class SpatialAttention(nn.Module):
    """CBAM-style spatial attention: channel-wise avg+max -> 7x7 conv -> sigmoid."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x):
        # x: (B, C, H, W)
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))   # (B, 1, H, W)
        return attn


class SCAFPN(nn.Module):
    """
    Spatial-Channel Attention FPN for MonoDETR.

    Input : list of N feature maps [(B, C, H_i, W_i)] in ascending stride order
            (e.g. 1/8, 1/16, 1/32, 1/64) - assumes already projected to common C.
    Output: list of N enhanced feature maps with the SAME shapes.

    Key design:
      * High-level (top) feature provides spatial guidance to lower levels.
      * Channel attention is applied to every level (cheap, level-specific).
      * Residual connections keep gradient flow healthy and let the model
        learn an identity mapping if the new module hurts at first.
    """

    def __init__(self, num_channels=256, num_levels=4, reduction=16,
                 spatial_kernel=7, use_high_guidance=True):
        super().__init__()
        self.num_levels = num_levels
        self.use_high_guidance = use_high_guidance

        # Per-level channel attention
        self.cas = nn.ModuleList([
            ChannelAttention(num_channels, reduction=reduction)
            for _ in range(num_levels)
        ])
        # Per-level local spatial attention
        self.sas = nn.ModuleList([
            SpatialAttention(kernel_size=spatial_kernel)
            for _ in range(num_levels)
        ])
        # A single high-level spatial attention for top-down guidance
        if use_high_guidance:
            self.high_sa = SpatialAttention(kernel_size=spatial_kernel)

        # Per-level lateral 1x1 conv to smooth the gated output
        self.laterals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=False),
                nn.GroupNorm(32, num_channels),
            )
            for _ in range(num_levels)
        ])

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, srcs):
        """
        Args:
            srcs (List[Tensor]): N feature maps after input_proj,
                ordered from highest-resolution to lowest (1/8 -> 1/64).
        Returns:
            List[Tensor]: enhanced features with identical shapes / order.
        """
        assert len(srcs) == self.num_levels, \
            f"SCA-FPN expects {self.num_levels} levels but got {len(srcs)}."

        # Step 1: channel attention per level (independent reweighting)
        x_ca = [src * self.cas[i](src) for i, src in enumerate(srcs)]

        # Step 2: compute high-level spatial attention from the TOPMOST level
        #         (lowest resolution = strongest semantics).
        if self.use_high_guidance:
            top_sa = self.high_sa(x_ca[-1])              # (B, 1, H_top, W_top)
        else:
            top_sa = None

        # Step 3: fuse local SA + upsampled top-down SA, apply gating, residual
        outs = []
        for i, feat_ca in enumerate(x_ca):
            local_sa = self.sas[i](feat_ca)              # (B, 1, H_i, W_i)
            if top_sa is not None and i != len(x_ca) - 1:
                # Upsample top-level SA to current resolution (linear is fine
                # for a 1-channel scalar map; nearest also works).
                guide_sa = F.interpolate(
                    top_sa, size=feat_ca.shape[-2:],
                    mode='bilinear', align_corners=False)
                # Element-wise mean of local + guidance gives the fused gate.
                fused_sa = 0.5 * (local_sa + guide_sa)
            else:
                # The top level uses only its own SA (it IS the guide).
                fused_sa = local_sa

            gated = feat_ca * fused_sa                    # spatial reweighting
            lat = self.laterals[i](gated)
            outs.append(lat + srcs[i])                    # residual to original

        return outs
