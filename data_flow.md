# MonoDETR 完整数据流 + P2-KHYP 改动点

> 基于 MonoDETR 论文（ICCV 2023）与源码，追踪从输入图像到最终损失的全链路 Shape 变化。
> 所有 Shape 基于 B=batch_size=16, H=384, W=1280, hidden_dim=256, K=4。

---

## 一、原始 MonoDETR 数据流

### 阶段1：Backbone（ResNet50）

```
输入: images [B, 3, 384, 1280]
  │
  ▼ ResNet50
  │
  ├─ features[0]: NestedTensor(tensors=[B, 256, 48, 160], mask=[B, 48, 160])  1/8 下采样
  ├─ features[1]: NestedTensor(tensors=[B, 512, 24, 80],  mask=[B, 24, 80])   1/16 下采样
  ├─ features[2]: NestedTensor(tensors=[B, 1024, 12, 40], mask=[B, 12, 40])   1/32 下采样
  └─ pos[0..2]: 对应各层的正弦位置编码，shape 同 features[i]
```

### 阶段2：Input Projection（1×1 Conv 通道对齐 → hidden_dim=256）

```
features[0] → input_proj[0]: Conv2d(256→256,1×1)+GN → srcs[0]: [B, 256, 48, 160]
features[1] → input_proj[1]: Conv2d(512→256,1×1)+GN → srcs[1]: [B, 256, 24, 80]
features[2] → input_proj[2]: Conv2d(1024→256,1×1)+GN→ srcs[2]: [B, 256, 12, 40]
features[2].tensors → input_proj[3]: Conv2d(256→256,3×3,s=2)+GN → srcs[3]: [B, 256, 6, 20]  (1/64 额外层)
```

`srcs = [srcs[0], srcs[1], srcs[2], srcs[3]]`，4 级多尺度特征。

### 阶段3：DepthPredictor（前景深度预测 + 深度位置编码）

```
输入: srcs(4级特征), masks[1], pos[1]
  │
  ├─ src_8  = downsample(srcs[0])   → [B, 256, 24, 80]
  ├─ src_16 = proj(srcs[1])         → [B, 256, 24, 80]
  ├─ src_32 = upsample(srcs[2])     → [B, 256, 24, 80]
  │
  ├─ src = (src_8 + src_16 + src_32) / 3     → [B, 256, 24, 80]  (朴素多尺度融合)
  ├─ depth_head: Conv→GN→ReLU→Conv→GN→ReLU   → [B, 256, 24, 80]
  ├─ depth_classifier: Conv2d(256→81, 1×1)   → [B, 81, 24, 80]  (80 bins + 末位填充)
  │                                                                   ↑ pred_depth_map_logits
  ├─ softmax + bin_center加权                → weighted_depth: [B, 24, 80]  (密集深度图)
  │
  ├─ depth_encoder(Transformer×1层)          → depth_pos_embed: [B, 256, 24, 80]  (深度特征)
  ├─ depth_pos_embed_ip = interpolate(weighted_depth, 61维embed)
  │                        → [B, 256, 24, 80]  (深度位置插值编码)
  └─ depth_pos_embed = depth_pos_embed + depth_pos_embed_ip → [B, 256, 24, 80]
```

返回值：
| 变量 | Shape | 含义 |
|---|---|---|
| `pred_depth_map_logits` | [B, 81, 24, 80] | 深度分类 logits |
| `depth_pos_embed` | [B, 256, 24, 80] | 深度特征 + 位置编码 |
| `weighted_depth` | [B, 24, 80] | 软分类加权深度图 |
| `depth_pos_embed_ip` | [B, 256, 24, 80] | 深度位置插值编码 |

### 阶段4：DepthAwareTransformer（Encoder + Decoder）

#### 4.1 Encoder 准备

```
4级 srcs 分别 flatten + level_embed 加和:
  srcs[0]: [B, 256, 48, 160] → flatten → [B, 7680, 256]
  srcs[1]: [B, 256, 24, 80]  → flatten → [B, 1920, 256]
  srcs[2]: [B, 256, 12, 40]  → flatten → [B, 480,  256]
  srcs[3]: [B, 256, 6,  20]  → flatten → [B, 120,  256]

cat → src_flatten: [B, 10200, 256]  (7680+1920+480+120)
mask_flatten:      [B, 10200]
spatial_shapes:    [[48,160], [24,80], [12,40], [6,20]]
```

#### 4.2 VisualEncoder（3 层 MSDeformAttn 多尺度自注意力）

```
输入: src_flatten [B, 10200, 256]
  │
  ├─ EncoderLayer×3: MSDeformAttn(self-attn, 4 levels, 8 heads, 4 pts) + FFN(256→256)
  └─ memory: [B, 10200, 256]  (Shape 不变)
```

#### 4.3 Decoder 准备

```
query_embeds: [550, 512]  (num_queries=50 × group_num=11, hidden_dim*2)
  ├─ 训练时: query_embeds = self.query_embed.weight → [550, 512]
  └─ 推理时: query_embeds = self.query_embed.weight[:50] → [50, 512]

→ expand为 [B, 550/50, 512]
→ 内部 split 为 tgt [B, NQ, 256] + reference_points [B, NQ, 2]
→ reference_points → sigmoid → init_reference [B, NQ, 2]
```

#### 4.4 DepthAwareDecoder（3 层，每层 4 个步骤）

```
输入: tgt [B, NQ, 256], init_reference [B, NQ, 2], memory [B, 10200, 256]
      depth_pos_embed [B, 256, 24, 80]
      depth_pos_embed_ip [B, 256, 24, 80]

每层 DecoderLayer:
  ┌─────────────────────────────────────────────────┐
  │ Step 1: Depth Cross-Attention                   │
  │   depth_pos_embed: flatten → [B, 1920, 256]    │
  │   tgt ← MultiheadAttention(Q=tgt, K/V=depth)    │
  │   输出: tgt [B, NQ, 256]                        │
  ├─────────────────────────────────────────────────┤
  │ Step 2: Self-Attention                          │
  │   query_pos = gen_sineembed(reference) [B,NQ,256] │
  │   tgt ← MultiheadAttention(Q=tgt+qpos, K=tgt+qpos, V=tgt)│
  │   输出: tgt [B, NQ, 256]                        │
  ├─────────────────────────────────────────────────┤
  │ Step 3: Deformable Cross-Attention              │
  │   tgt ← MSDeformAttn(Q=tgt+qpos, ref=reference, src=memory)│
  │   输出: tgt [B, NQ, 256]                        │
  ├─────────────────────────────────────────────────┤
  │ Step 4: FFN                                     │
  │   tgt ← Linear→ReLU→Linear(tgt)                 │
  │   输出: tgt [B, NQ, 256]                        │
  │                                                 │
  │ Box Refine: bbox_embed(tgt) + inv_sig(reference)│
  │   → sigmoid → new_reference [B, NQ, 6]          │
  │ dim_embed(tgt) → 3D尺寸 [B, NQ, 3]              │
  └─────────────────────────────────────────────────┘

3 层 decoder 输出:
  hs:                  [3, B, NQ, 256]   ← 各层 tgt
  init_reference:      [B, NQ, 2]
  inter_references:    [3, B, NQ, 6]     ← 各层 refine 后的 reference
  inter_references_dim: [3, B, NQ, 3]    ← 各层 3D 尺寸
```

### 阶段5：预测头（逐层、逐 query 并行）

```
对每层 lvl∈{0,1,2}:
  hs[lvl]: [B, NQ, 256]

  ├─ class_embed[lvl]: Linear(256→3)              → outputs_class:  [B, NQ, 3]
  ├─ bbox_embed[lvl]: MLP(256→256→6) + reference  → outputs_coord:  [B, NQ, 6]  (cx,cy,l,r,t,b 归一化)
  ├─ dim_embed_3d[lvl]  (已在 decoder 内调用)       → size3d:         [B, NQ, 3]  (h,w,l 残差)
  ├─ angle_embed[lvl]: MLP(256→256→24)            → outputs_angle:  [B, NQ, 24] (12bin+12res)
  └─ depth_embed[lvl]: MLP(256→256→2)             → depth_reg:      [B, NQ, 2]  (mean, log_var)
```

### 阶段6：三源深度融合（原始 MonoDETR 核心）

```
从 depth_reg [B, NQ, 2] 提取:
  d_reg = 1/(sigmoid(depth_reg[:,:,0]) + ε) - 1   → [B, NQ]  (逆 sigmoid 到实数深度)

从几何约束:
  d_geo = size3d[:,:,0] / box2d_height × fx        → [B, NQ]  (f·H/h)

从深度图采样:
  d_map = grid_sample(weighted_depth, center3d_2d)  → [B, NQ]  (双线性插值)

融合:
  depth_ave = cat([ (d_reg + d_geo + d_map)/3,      → [B, NQ, 1]
                    depth_reg[:,:,1:2] ])             → [B, NQ, 1]
            → [B, NQ, 2]  (融合深度, log_variance)
```

### 阶段7：输出整合

```
最终输出 out:
  pred_logits:          outputs_class[-1]     [B, NQ, 3]
  pred_boxes:           outputs_coord[-1]     [B, NQ, 6]
  pred_3d_dim:          inter_references_dim[-1] [B, NQ, 3]
  pred_depth:           outputs_depth[-1]     [B, NQ, 2]
  pred_angle:           outputs_angle[-1]     [B, NQ, 24]
  pred_depth_map_logits:[B, 81, 24, 80]

aux_outputs (2 层中间层):
  [{pred_logits, pred_boxes, pred_3d_dim, pred_angle, pred_depth}, ...] ×2
```

### 阶段8：SetCriterion（损失函数）

```
1. HungarianMatcher: 匈牙利算法匹配预测↔GT
   代价矩阵 = cost_class(2×) + cost_bbox(5×) + cost_giou(2×) + cost_3dcenter(10×)
   输出: indices = [(src_idx, tgt_idx) × B]
   num_boxes = Σ len(t["labels"]) × group_num(11)

2. 各损失:
   loss_labels:    sigmoid_focal_loss(logits, onehot GT)
   loss_boxes:     L1(l,r,t,b) + GIoU
   loss_3dcenter:  L1(cx3d, cy3d)
   loss_depths:    1.4142×exp(-log_var)×|depth_ave - GT_depth| + log_var   ← 拉普拉斯NLL
   loss_dims:      dim_aware_L1
   loss_angles:    CrossEntropy(12bin) + L1(residual)
   loss_depth_map: DDNLoss(focal, 前后景平衡)

3. aux loss: 中间层重复步骤1-2，depth_map损失仅最后一层计算
```

---

## 二、P2-KHYP 改进后的数据流

### 改动点定位

P2 改进集中在两个地方：

| 位置 | 原始 | P2-KHYP 改动 |
|---|---|---|
| **深度头输出** | `depth_embed: MLP(256→256→2)` | `depth_embed: MLP(256→256→K×2)` |
| **新增结构** | 无 | `hypo_conf_embed: MLP(256→256→K)` |
| **深度融合** | 单点 `(d_reg+d_geo+d_map)/3` | K 假设独立融合 |
| **深度损失** | 单点拉普拉斯 NLL | WTA + diversity loss |

**不改的地方**：Backbone、input_proj、DepthPredictor、Transformer Encoder/Decoder、class_embed、bbox_embed、dim_embed_3d、angle_embed 全部不动。

### 改动后的数据流（只在阶段5-6-8变化）

#### 改动后阶段5：深度头

```
原始:
  depth_embed[lvl]: MLP(256→256→2) → depth_reg: [B, NQ, 2]

改为:
  depth_embed[lvl]: MLP(256→256→K×2) → depth_reg: [B, NQ, K×2]  (K=4 → 8维)
  hypo_conf_embed[lvl]: MLP(256→256→K) → hypo_conf_logits: [B, NQ, K]  (新增)
```

#### 改动后阶段6：K 假设深度融合

```
depth_reg: [B, NQ, K×2]
  → view(B, NQ, K, 2)

提取:
  depth_hyps = 1/(sigmoid(depth_reg[...,0]) + ε) - 1  → [B, NQ, K]  (K个回归深度)
  log_vars   = depth_reg[...,1]                        → [B, NQ, K]  (K个方差)

三源融合 (每个假设独立):
  depth_fused = (depth_hyps + d_geo.unsqueeze(-1) + d_map.unsqueeze(-1)) / 3
              → [B, NQ, K]  ← K个融合深度

置信度加权平均 (用于下游兼容):
  hypo_conf = softmax(hypo_conf_logits, dim=-1)        → [B, NQ, K]
  depth_ave_val = (hypo_conf × depth_fused).sum(-1)    → [B, NQ]
  log_var_ave   = (hypo_conf × log_vars).sum(-1)       → [B, NQ]
  depth_ave = stack([depth_ave_val, log_var_ave], -1)  → [B, NQ, 2]  ← shape兼容原始
```

`d_geo` 和 `d_map` 各为一个值 [B, NQ]，对所有 K 个假设贡献相同偏置。假设间差异完全来自回归头：

```
depth_fused_i - depth_fused_j = (depth_hyps_i - depth_hyps_j) / 3
```

证明了融合不破坏多样性。

#### 改动后输出 dict 新增字段

```
原始 out:
  pred_depth: [B, NQ, 2]  (兼容接口，不变)

P2 新增:
  depth_fused: [3, B, NQ, K]   ← K=4 个融合深度，3层decoder
  log_vars:    [3, B, NQ, K]   ← K=4 个方差
  (aux_outputs 同理携带)
```

#### 改动后阶段8：深度损失

```
原始:
  src_depths = pred_depth[idx]              → [M, 2]
  loss = LaplacianNLL(depth_input, log_var, GT)  → 标量

P2:
  depth_fused = out['depth_fused'][idx]     → [M, K=4]
  log_vars    = out['log_vars'][idx]        → [M, K=4]

  Step 1 - Winner-Take-All:
    errors = |depth_fused - GT.unsqueeze(-1)|  → [M, K]
    if ε-greedy(10%): best_k = randint(0,K)
    else:             best_k = argmin(errors)   → [M]
    gather best_k → depth_pred_best [M], log_var_best [M]

  Step 2 - 深度损失 (仅best假设):
    loss_depth = 1.4142×exp(-log_var_best)×|depth_pred_best-GT| + log_var_best

  Step 3 - 多样性损失 (新增):
    pairwise_diff = |depth_fused.unsqueeze(-1) - depth_fused.unsqueeze(-2)|  → [M, K, K]
    loss_diversity = -mean(pairwise_diff的off-diagonal) × weight

  total: loss_depth + loss_depth_diversity
```

---

## 三、两部分数据流对比总图

```
                    原始 MonoDETR                          P2-KHYP 改动
                    ═══════════                          ═════════
images [B,3,384,1280]
        │
   ┌────▼────┐
   │ Backbone │  ResNet50
   └────┬────┘
features[0..2]+pos[0..2]
        │
   ┌────▼────┐
   │input_proj│  1×1 Conv → 256ch
   └────┬────┘
srcs[0..3]  (4 levels)
        │
   ┌────▼──────────┐
   │DepthPredictor  │  深度图 + 位置编码
   └────┬──────────┘
depth_pos_embed, weighted_depth, pred_depth_map_logits
        │
   ┌────▼──────────────┐
   │DepthAwareTransformer│  Encoder×3 + Decoder×3
   └────┬──────────────┘
hs[3,B,NQ,256], references, sizes
        │
   ┌────▼────┐                              ┌────▼──────────┐
   │预测头并行│  ← 唯一改动点 →               │预测头并行     │
   │         │                              │               │
   │class: 3 │                              │class: 3 (不变)│
   │bbox: 6  │                              │bbox: 6  (不变)│
   │dim: 3   │                              │dim: 3   (不变)│
   │angle:24 │                              │angle:24 (不变)│
   │depth: 2 │ MLP(256→2)                  │depth:K×2│MLP(256→K×2)│
   │         │                              │conf: K  │MLP(256→K) │ NEW!
   └────┬────┘                              └────┬──────────┘
        │                                        │
   ┌────▼────┐  单点融合                  ┌──────▼──────┐  K假设融合
   │d_reg    │  [B,NQ]                   │depth_hyps   │  [B,NQ,K]
   │+ d_geo  │  [B,NQ]                   │+ d_geo      │  [B,NQ,1]  broadcast
   │+ d_map  │  [B,NQ]                   │+ d_map      │  [B,NQ,1]  broadcast
   │─────────│                           │─────────────│
   │depth_ave│  [B,NQ,2]                 │depth_fused  │  [B,NQ,K]
   └────┬────┘                           │depth_ave    │  [B,NQ,2]  置信度加权
        │                                └──────┬──────┘
        │                                        │
   ┌────▼────┐  损失                      ┌──────▼──────┐  损失
   │Laplacian│  单点NLL                  │WTA+ε-greedy│  多假设选择
   │NLL      │                           │+ diversity │  鼓励分化
   └─────────┘                           └─────────────┘
```

---

## 四、改动文件与代码行号对照

| 文件 | 改动位置 | 改动内容 |
|---|---|---|
| `configs/monodetr.yaml` | L49-56 | 新增 `use_k_hypothesis`, `num_hypotheses`, `hypo_diversity_weight`, `hypo_epsilon` |
| `monodetr.py` `__init__` | L31-32,63-69,156-158 | 新增参数；depth_embed→K×2；新增hypo_conf_embed |
| `monodetr.py` `forward` | L221-223,268-290 | K假设三源融合；depth_ave置信度加权 |
| `monodetr.py` `_set_aux_loss` | L324-340 | 传递depth_fused/log_vars到aux层 |
| `monodetr.py` `SetCriterion.__init__` | L349-371 | 新增K假设相关参数 |
| `monodetr.py` `loss_depths` | L446-485 | WTA(ε-greedy) + diversity loss；else分支原代码 |
| `monodetr.py` `build()` | L641-656,670-671,691-701 | cfg.get()透传；weight_dict新增loss_depth_diversity |

---

## 五、推理时的数据流

推理时设置 `use_k_hypothesis=True`，`forward` 中的 `depth_ave` 是置信度加权平均 `[B, NQ, 2]`，与原始 shape 完全一致。`decode_helper.py` 中：

```python
depth = outputs['pred_depth'][:, :, 0: 1]   # → [B, NQ, 1]  置信度加权的融合深度
sigma = outputs['pred_depth'][:, :, 1: 2]   # → [B, NQ, 1]  置信度加权的 log_var
sigma = torch.exp(-sigma)                    # → 标准差
```

推理路径无需任何改动，因为置信度加权自动完成了"选择最优假设"的操作。
