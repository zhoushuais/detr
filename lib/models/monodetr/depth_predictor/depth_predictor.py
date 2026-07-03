import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import TransformerEncoder, TransformerEncoderLayer


class DepthPredictor(nn.Module):

    def __init__(self, model_cfg):
        """
        Initialize depth predictor and depth encoder
        Args:
            model_cfg [EasyDict]: Depth classification network config
        """
        super().__init__()
        # [P2-DBDU] Dynamic Bins switch. When enabled, the bin centers are
        # predicted per-image (AdaBins style) instead of the fixed LID partition.
        self.use_dynamic_bins = bool(model_cfg.get("use_dynamic_bins", False))
        depth_num_bins = int(model_cfg["num_depth_bins"])
        if self.use_dynamic_bins:
            depth_num_bins = int(model_cfg.get("num_dynamic_bins", depth_num_bins))
        depth_min = float(model_cfg["depth_min"])
        depth_max = float(model_cfg["depth_max"])
        self.depth_max = depth_max
        # [P2-DBDU] kept for the dynamic-bin forward / loss path
        self.depth_min = depth_min
        self.depth_num_bins = depth_num_bins

        bin_size = 2 * (depth_max - depth_min) / (depth_num_bins * (1 + depth_num_bins))
        bin_indice = torch.linspace(0, depth_num_bins - 1, depth_num_bins)
        bin_value = (bin_indice + 0.5).pow(2) * bin_size / 2 - bin_size / 8 + depth_min
        bin_value = torch.cat([bin_value, torch.tensor([depth_max])], dim=0)
        self.depth_bin_values = nn.Parameter(bin_value, requires_grad=False)

        # Create modules
        d_model = model_cfg["hidden_dim"]
        self.downsample = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.GroupNorm(32, d_model))
        self.proj = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(1, 1)),
            nn.GroupNorm(32, d_model))
        self.upsample = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(1, 1)),
            nn.GroupNorm(32, d_model))

        self.depth_head = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=d_model),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=d_model),
            nn.ReLU())

        self.depth_classifier = nn.Conv2d(d_model, depth_num_bins + 1, kernel_size=(1, 1))

        depth_encoder_layer = TransformerEncoderLayer(
            d_model, nhead=8, dim_feedforward=256, dropout=0.1)

        self.depth_encoder = TransformerEncoder(depth_encoder_layer, 1)

        self.depth_pos_embed = nn.Embedding(int(self.depth_max) + 1, 256)

        # [P2-DBDU] AdaBins-style per-image bin predictor: pooled depth feature
        # -> per-image BOUNDED perturbation around the LID prior -> edges/centers.
        #
        # Earlier design used a free softmax over raw widths. Diagnostics showed
        # it COLLAPSED: ~60/80 bins crushed into the [56,60]m far sliver, leaving
        # the useful 0-56m range only ~20 effective bins (drift from LID ~31m),
        # which is strictly coarser than fixed LID and explained the regression.
        # Fix: keep the fixed LID log-widths as a prior buffer and let the head
        # output only a tanh-bounded perturbation (|delta| <= adapt_scale per
        # logit). At iter 0 (zero-init weight AND bias) delta=0 -> exact LID; the
        # bound caps each bin width within ~e^±adapt_scale of LID, so the bins
        # can adapt per-image but can no longer collapse away 60 bins.
        if self.use_dynamic_bins:
            self.bins_adapt_scale = float(model_cfg.get("dynamic_bins_adapt_scale", 0.5))
            self.bin_predictor = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, depth_num_bins))
            lid_widths = torch.arange(1, depth_num_bins + 1, dtype=torch.float32)
            lid_log = torch.log(lid_widths / lid_widths.sum())
            self.register_buffer("lid_log_widths", lid_log)   # fixed LID prior
            nn.init.zeros_(self.bin_predictor[-1].weight)
            nn.init.zeros_(self.bin_predictor[-1].bias)

    def predict_dynamic_bins(self, src):
        # [P2-DBDU] src: (B, C, H, W) after depth_head -> per-image bin partition.
        B = src.shape[0]
        global_feat = F.adaptive_avg_pool2d(src, 1).flatten(1)             # (B, C)
        # bounded per-image perturbation around the fixed LID log-widths prior:
        # delta in [-adapt_scale, +adapt_scale] per logit (tanh) -> bins stay
        # within ~e^±adapt_scale of LID after softmax, can't collapse.
        delta = torch.tanh(self.bin_predictor(global_feat)) * self.bins_adapt_scale
        widths = F.softmax(self.lid_log_widths + delta, dim=-1)            # (B, num_bins), sum=1
        widths = widths * (self.depth_max - self.depth_min)               # actual widths
        edges_inner = self.depth_min + torch.cumsum(widths, dim=-1)        # (B, num_bins), last == depth_max
        left = torch.full((B, 1), self.depth_min, device=src.device, dtype=edges_inner.dtype)
        edges = torch.cat([left, edges_inner], dim=-1)                     # (B, num_bins+1)
        centers = (edges[:, :-1] + edges[:, 1:]) / 2                       # (B, num_bins) midpoints
        last = torch.full((B, 1), self.depth_max, device=src.device, dtype=centers.dtype)
        centers = torch.cat([centers, last], dim=-1)                      # (B, num_bins+1) to match classifier
        return centers, edges

    def forward(self, feature, mask, pos):
       
        assert len(feature) == 4

        # foreground depth map
        #ipdb.set_trace()
        src_16 = self.proj(feature[1])
        src_32 = self.upsample(F.interpolate(feature[2], size=src_16.shape[-2:], mode='bilinear'))
        src_8 = self.downsample(feature[0])
        #new_add
        # src_8 = self.proj(feature[0])
        # src_16 = self.upsample(F.interpolate(feature[1], size=src_8.shape[-2:], mode='bilinear'))
        # src_32 = self.upsample(F.interpolate(feature[2], size=src_8.shape[-2:], mode='bilinear'))
        ####
        src = (src_8 + src_16 + src_32) / 3

        src = self.depth_head(src)
        #ipdb.set_trace()
        depth_logits = self.depth_classifier(src)

        depth_probs = F.softmax(depth_logits, dim=1)
        # [P2-DBDU] dynamic per-image bins vs fixed LID bins
        if self.use_dynamic_bins:
            bin_centers, bin_edges = self.predict_dynamic_bins(src)
            weighted_depth = (depth_probs * bin_centers[:, :, None, None]).sum(dim=1)
        else:
            bin_edges = None
            weighted_depth = (depth_probs * self.depth_bin_values.reshape(1, -1, 1, 1)).sum(dim=1)
        #ipdb.set_trace()
        # depth embeddings with depth positional encodings
        B, C, H, W = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        pos = pos.flatten(2).permute(2, 0, 1)

        depth_embed = self.depth_encoder(src, mask, pos)
        depth_embed = depth_embed.permute(1, 2, 0).reshape(B, C, H, W)
        #ipdb.set_trace()
        depth_pos_embed_ip = self.interpolate_depth_embed(weighted_depth)
        depth_embed = depth_embed + depth_pos_embed_ip

        # [P2-DBDU] bin_edges (B, num_bins+1) is None in baseline mode; the depth
        # map loss uses it to bucketize GT depths against the per-image partition.
        return depth_logits, depth_embed, weighted_depth, depth_pos_embed_ip, bin_edges

    def interpolate_depth_embed(self, depth):
        depth = depth.clamp(min=0, max=self.depth_max)
        pos = self.interpolate_1d(depth, self.depth_pos_embed)
        pos = pos.permute(0, 3, 1, 2)
        return pos

    def interpolate_1d(self, coord, embed):
        floor_coord = coord.floor()
        delta = (coord - floor_coord).unsqueeze(-1)
        floor_coord = floor_coord.long()
        ceil_coord = (floor_coord + 1).clamp(max=embed.num_embeddings - 1)
        return embed(floor_coord) * (1 - delta) + embed(ceil_coord) * delta
