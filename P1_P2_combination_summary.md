# P1 + P2 组合实现总结

当前组合分支：`p1-p2-combined`

组合目标：将 `p1-sca-fpn` 分支中的 SCA-FPN 特征增强模块，与 `P2` 分支中的 Depth-only CoP 深度预测头组合为一套完整训练代码，用于后续 `P1 + P2` 联合实验。

---

## 1. P1 分支改进内容

来源分支：`p1-sca-fpn`

核心代码文件：

- `lib/models/monodetr/sca_fpn.py`
- `lib/models/monodetr/monodetr.py`
- `configs/monodetr.yaml`

### 1.1 改进位置

P1 作用于 MonoDETR 的多尺度图像特征阶段。具体插入位置为：

```text
Backbone
  -> input_proj
  -> P1: SCA-FPN
  -> DepthPredictor
  -> Depth-aware Transformer
```

也就是说，P1 在 `input_proj` 已经把不同 backbone level 统一到 `hidden_dim=256` 之后，对 `srcs` 进行增强。增强后的 `srcs` 同时送入：

- `depth_predictor`
- `depthaware_transformer`

因此 P1 会同时影响密集深度预测和 transformer query 特征。

### 1.2 模块结构

SCA-FPN 包含：

1. Channel Attention  
   每个 feature level 独立使用 CBAM-style 通道注意力。

2. Spatial Attention  
   每个 feature level 计算局部空间注意力。

3. High-level Guidance  
   使用最高层特征的空间注意力上采样指导低层特征，给小目标所在的高分辨率层注入更强语义线索。

4. Optional DCN Lateral  
   lateral path 可使用 3x3 DeformConv2d，offset conv 零初始化，使训练初期接近普通卷积。

5. Residual Connection  
   输出形式为：

```text
out_i = lateral(gated_i) + src_i
```

残差连接用于降低训练扰动。

### 1.3 P1 的论文定位

P1 属于：

```text
特征增强 / neck 改进
```

它主要解决：

```text
小目标在多尺度特征中语义弱、分辨率低、容易被背景淹没的问题。
```

---

## 2. P2 分支改进内容

来源分支：`P2`

核心代码文件：

- `lib/models/monodetr/monodetr.py`
- `configs/monodetr.yaml`

### 2.1 改进位置

P2 作用于 decoder 输出后的属性预测头阶段。具体位置为：

```text
Depth-aware Transformer decoder
  -> query feature hs
  -> P2: Depth-only CoP depth head
  -> pred_depth
```

### 2.2 模块结构

P2 新增 `CoPDepthHead`：

```text
dim_feat   = MLP(hs)
angle_feat = MLP(concat(hs, dim_feat))
depth_feat = MLP(concat(hs, dim_feat, angle_feat))
depth_feat = depth_feat + dim_feat + angle_feat
depth_reg  = Linear(depth_feat)
```

其中 `dim_feat` 和 `angle_feat` 只作为 depth head 内部条件特征，不直接替代最终监督输出。

### 2.3 Depth-only 设计

当前主方案为：

```yaml
use_cop: True
cop_mode: 'depth_only'
```

该模式下：

- `pred_3d_dim` 仍来自 MonoDETR 原始 `inter_references_dim`
- `pred_angle` 仍来自 MonoDETR 原始 `angle_embed(hs)`
- 只有 `depth_reg` 改为由 `CoPDepthHead(hs)` 生成
- `pred_depth` 的输出形状仍为 `[B, NQ, 2]`
- loss、matcher、decode 均不改

### 2.4 P2 的论文定位

P2 属于：

```text
预测头增强 / depth head 改进
```

它主要解决：

```text
单目 3D 小目标 query-level 深度预测不稳定的问题。
```

---

## 3. P1 与 P2 的组合逻辑

P1 和 P2 作用于 MonoDETR 的不同阶段：

| 模块 | 作用位置 | 输入 | 输出 | 解决问题 |
|---|---|---|---|---|
| P1 SCA-FPN | Transformer 前 | 多尺度 `srcs` | 增强后的 `srcs` | 小目标特征弱 |
| P2 Depth-only CoP | Decoder 后 | query feature `hs` | 增强后的 `depth_reg` | 深度预测不稳 |

组合后的完整数据流为：

```text
Input image
  -> Backbone
  -> input_proj
  -> P1: SCA-FPN
  -> DepthPredictor
  -> Depth-aware Transformer
  -> P2: Depth-only CoP depth head
  -> three-source depth fusion
  -> 3D detection outputs
```

两者不冲突的原因：

1. P1 改的是 transformer 输入特征。
2. P2 改的是 decoder 输出后的 depth head。
3. P1 不改变 `hs` 的 shape，只改变特征值。
4. P2 不改变 `pred_depth` 接口，只改变 `depth_reg` 的生成方式。
5. loss、matcher、decode 不需要额外改动。

---

## 4. 当前组合分支代码改动

当前 `p1-p2-combined` 分支只引入训练必要代码：

| 文件 | 改动 |
|---|---|
| `lib/models/monodetr/sca_fpn.py` | 从 P1 分支引入 SCA-FPN 模块 |
| `lib/models/monodetr/monodetr.py` | 同时接入 SCA-FPN 和 Depth-only CoP |
| `configs/monodetr.yaml` | 默认开启 P1 + P2 组合训练配置 |

没有合并 P1 分支中的 PDF、调研文档、workflow 文档等非训练文件。

---

## 5. 当前默认训练配置

组合分支当前默认配置为：

```yaml
use_sca_fpn: True
sca_fpn_reduction: 16
sca_fpn_kernel: 7
sca_fpn_high_guidance: True
use_dcn_lateral: True
dcn_kernel: 3

use_cop: True
cop_mode: 'depth_only'
```

数据类别保持为：

```yaml
writelist: ['Car', 'Pedestrian', 'Cyclist']
```

这代表当前 `configs/monodetr.yaml` 可以直接用于 `P1 + P2` 组合实验。

---

## 6. 建议实验顺序

后续建议按以下顺序跑：

| 实验 | 配置 | 目的 |
|---|---|---|
| Baseline | `use_sca_fpn=False`, `use_cop=False` | 原始对照 |
| P1 only | `use_sca_fpn=True`, `use_cop=False` | 验证特征增强 |
| P2 only | `use_sca_fpn=False`, `use_cop=True` | 验证深度头增强 |
| P1 + P2 | `use_sca_fpn=True`, `use_cop=True` | 验证最终组合方法 |

当前组合分支默认就是第四行。

建议 `P1 + P2` 至少跑三次，继续按当前规则：

```text
checkpoint_best.pth = 按 (Car_3d_moderate_R40 + Pedestrian_3d_moderate_R40 + Cyclist_3d_moderate_R40) / 3 选 best
```

然后与三次 baseline、三次 P1、三次 P2 做 mean±std 对比。

---

## 7. 论文中的组合叙事

如果 `P1 + P2` 实验稳定，论文主方法可以概括为：

> 本文从特征表达和深度预测两个层面对 MonoDETR 进行改进。首先，设计 SCA-FPN 模块对多尺度图像特征进行空间-通道增强，并利用高层语义注意力指导低层小目标特征；其次，设计属性条件引导的 Depth-only CoP 深度预测头，在 query-level 深度回归中引入尺寸和朝向相关条件特征。前者提升小目标特征可见性，后者增强小目标深度估计稳定性，二者共同改善单目 3D 小目标检测性能。
