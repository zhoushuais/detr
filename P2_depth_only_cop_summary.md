# P2 改进总结：属性条件引导的 Depth-only CoP 深度预测头增强

## 1. 研究定位

本项目的毕业论文主题是面向自动驾驶场景的单目 3D 小目标检测。P2 方向不是轻量化，也不是单纯增加网络复杂度，而是围绕单目 3D 检测中的核心难点进行精度改进：小目标在图像中占据像素少，视觉几何线索弱，尤其难以稳定预测 3D 深度。

MonoDETR 作为 baseline，已经通过 depth-aware transformer、密集深度图预测和三源深度融合引入了深度感知能力。但从当前实验和代码分析看，最终 3D 框深度仍然强依赖每个 object query 的 `depth_embed` 回归头。因此，P2 的核心定位是：

> 在不破坏 MonoDETR 原有检测框架的前提下，增强 per-query 深度预测头，使其更充分利用目标属性条件信息，从而提升小目标 3D 检测精度。

P2 最终建议名称可以写为：

> 属性条件引导的深度预测头增强模块

英文可写为：

> Attribute-conditioned Depth Prediction Head

如果希望保留 CoP 的术语，也可以写为：

> Depth-only Chain-of-Prediction Head

---

## 2. Baseline 问题分析

### 2.1 MonoDETR 的深度预测路径

MonoDETR 的整体深度相关路径可以概括为：

1. Backbone 提取多尺度图像特征。
2. DepthPredictor 预测前景密集深度图，得到 `weighted_depth`。
3. Depth-aware Transformer 使用深度位置编码增强 query 与图像特征交互。
4. Decoder 输出每个 object query 的特征 `hs`。
5. 预测头分别输出类别、2D/3D 框、尺寸、朝向和深度。
6. 最终深度由三源融合得到：
   - per-query 回归深度 `d_reg`
   - 几何深度 `d_geo`
   - 深度图采样值 `d_map`

原始 MonoDETR 中的深度融合形式为：

```text
d_final = (d_reg + d_geo + d_map) / 3
```

其中 `d_reg` 来自 `depth_embed(hs)`，是每个 query 自身预测的深度。虽然 `d_geo` 和 `d_map` 也参与融合，但实际实验说明，若只改动分桶或三源加权方式，最终 AP 很难稳定提升。这说明 per-query depth head 是最终深度质量的关键瓶颈之一。

### 2.2 小目标检测中的深度困难

小目标，尤其是 Pedestrian 和 Cyclist，在 KITTI 图像中通常具有以下特点：

- 图像占据区域小，纹理和边界信息有限。
- 遮挡、截断和远距离样本更多。
- 2D 框高度变化对深度估计非常敏感。
- 单目条件下，同一个 2D 外观可能对应多个合理 3D 深度。

因此，小目标的 3D 检测通常不是简单的分类问题，而是深度和几何属性估计问题。对于 Pedestrian 和 Cyclist，真实 3D 尺寸、朝向和深度之间存在较强耦合关系。如果深度头只从 query feature 独立回归深度，就容易忽略这种属性间约束。

---

## 3. P2 前期负结果与经验教训

P2 早期尝试过两个方向：Dynamic Bins 和几何深度不确定性加权融合。这两个方向均未带来稳定正增益。

### 3.1 Dynamic Bins

Dynamic Bins 的思路是把固定 LID 深度分桶改成每张图自适应的动态分桶。其直觉是让深度分类空间更贴合当前图像的目标分布。

但实验发现：

- 自由 softmax 版本容易出现 bin collapse，大量深度桶挤在相近距离。
- 有界 tanh 版本虽然缓解坍缩，但容易退化回全局静态分桶。
- AP 基本落在 baseline 噪声范围内，没有稳定提升。

根因是：Dynamic Bins 主要影响 `weighted_depth` 和 depth positional embedding，而最终 3D 框深度仍由 per-query `depth_embed` 主导。它没有直接约束最终 query-level depth prediction。

### 3.2 几何深度不确定性加权融合

该方向尝试将原始三源平均融合：

```text
(d_reg + d_geo + d_map) / 3
```

替换为基于不确定性的加权融合。

但实验发现：

- 无安全门控版本会导致检测器性能明显崩溃。
- 零初始化门控版本较安全，但容易退化回 baseline。
- 即使采用更稳定的 clamp-linear gate，AP 仍处于 baseline 噪声范围内。

根因是：该方法只是重新分配已有三个深度源的权重。如果三个源本身没有产生新的有效深度表达，仅靠加权难以创造新的深度信息。

### 3.3 得到的关键教训

早期 P2 实验说明：

> 只调整深度图、分桶或三源融合权重是不够的。若要真正改善最终 3D 框深度，必须直接增强 per-query depth head。

这也是后续从 DBDU 转向 CoP-style depth head 的原因。

---

## 4. 方法动机：为什么使用属性条件引导

单目 3D 检测中的深度不是孤立变量。对于一个目标，其深度估计通常与以下属性相关：

- 2D 框高度
- 3D 尺寸
- 朝向角
- 类别形状先验
- 图像局部纹理和遮挡状态

例如，同样的 2D 框高度，如果目标真实高度不同，则对应深度也不同；同样的可见区域，如果目标朝向不同，2D 投影形状也会不同。因此，尺寸和朝向信息可以作为深度估计的条件线索。

原始 MonoDETR 中，`dim_embed_3d`、`angle_embed` 和 `depth_embed` 主要是并行预测头。它们共享 decoder query feature，但预测过程本身缺少显式属性传播路径。

P2 的核心想法是引入链式属性条件建模：

```text
query feature -> dim feature -> angle feature -> depth feature -> depth_reg
```

其中 `dim_feat` 和 `angle_feat` 不一定直接替代原模型的尺寸和朝向输出，而是作为深度预测的条件特征，使 depth head 获得更强的属性感知能力。

---

## 5. 尝试一：Full CoP Replacement

### 5.1 方法设计

最初实现的 Full CoP replacement 将 `dim/angle/depth` 三个预测头全部纳入链式预测路径：

```text
hs -> dim_feat -> pred_dim
concat(hs, dim_feat) -> angle_feat -> pred_angle
concat(hs, dim_feat, angle_feat) -> depth_feat -> pred_depth
```

也就是说，`pred_3d_dim`、`pred_angle` 和 `pred_depth` 都由 CoP 特征链生成。

该版本的动机是尽可能贴近 Chain-of-Prediction 的思想，让 3D 属性之间形成显式传播关系。

### 5.2 实验结果

三次 Full CoP replacement 与三次 baseline 的均值如下：

| 方法 | Car Mod | Car Hard | Ped Mod | Ped Hard | Cyc Mod | Cyc Hard |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 19.3613±0.6280 | 16.3624±0.8570 | 5.4446±0.3652 | 4.3327±0.2982 | 4.4688±0.6479 | 4.1673±0.5800 |
| Full CoP replacement | 18.9896±0.9216 | 15.9353±0.7208 | 6.2343±0.6636 | 4.8963±0.5455 | 4.0119±0.9861 | 3.6968±0.7612 |
| 差值 | -0.3716 | -0.4271 | +0.7897 | +0.5636 | -0.4568 | -0.4705 |

### 5.3 结果分析

Full CoP replacement 对 Pedestrian 有稳定提升，但 Car 和 Cyclist 均值下降。这说明链式属性建模确实对行人类目标有帮助，但直接替换全部属性预测头会破坏 MonoDETR 原有尺寸和朝向预测路径的稳定性。

这次消融得到的重要结论是：

> CoP 的属性条件思想是有价值的，但不应该粗暴替换全部属性头。更合理的做法是在保留 MonoDETR 原有 dim/angle 稳定路径的基础上，只增强最终 depth head。

因此，Full CoP replacement 不作为最终 P2 方案，而作为消融实验保留。

---

## 6. 最终方法：Depth-only CoP

### 6.1 方法思想

Depth-only CoP 的设计原则是：

1. 保留 MonoDETR 原有的尺寸预测路径 `inter_references_dim`。
2. 保留 MonoDETR 原有的朝向预测路径 `angle_embed(hs)`。
3. 仅将原始 `depth_embed(hs)` 替换为属性条件引导的 CoP depth head。

也就是说，CoP 特征链只负责生成 per-query `depth_reg`，不直接替代最终 `pred_3d_dim` 和 `pred_angle`。

这种设计更稳健，因为它避免破坏原始模型中已经训练稳定的 dim/angle 分支，同时仍然让 depth head 获得属性条件信息。

### 6.2 网络结构

Depth-only CoP 的内部结构为：

```text
dim_feat   = MLP_dim(hs)
angle_feat = MLP_angle(concat(hs, dim_feat))
depth_feat = MLP_depth(concat(hs, dim_feat, angle_feat))
depth_feat = depth_feat + dim_feat + angle_feat
depth_reg  = Linear(depth_feat)
```

其中：

- `hs` 是 decoder 输出的 query feature。
- `dim_feat` 是尺寸相关条件特征，不直接作为最终尺寸输出。
- `angle_feat` 是朝向相关条件特征，不直接作为最终朝向输出。
- `depth_feat` 聚合 query、尺寸条件和朝向条件。
- `depth_reg` 输出 `[depth_logit, log_variance]`，保持与原始 MonoDETR 的 `pred_depth` 接口一致。

### 6.3 与原始 MonoDETR 的关系

Depth-only CoP 不改变以下接口：

- `pred_logits`
- `pred_boxes`
- `pred_3d_dim`
- `pred_angle`
- `pred_depth`
- `aux_outputs`
- `decode_helper.py`
- `SetCriterion.loss_depths`

它只改变 `depth_reg` 的生成方式。

原始 MonoDETR：

```text
depth_reg = depth_embed(hs)
```

Depth-only CoP：

```text
dim_feat, angle_feat, depth_reg = CoPDepthHead(hs)
```

随后仍然使用原始三源融合：

```text
d_final = (d_reg + d_geo + d_map) / 3
```

其中：

```text
d_reg = 1 / sigmoid(depth_reg[0]) - 1
```

最终 `pred_depth` 仍为：

```text
pred_depth = [d_final, log_variance]
```

### 6.4 代码实现位置

主要改动位于：

- `lib/models/monodetr/monodetr.py`
- `configs/monodetr.yaml`

关键配置为：

```yaml
use_cop: True
cop_mode: 'depth_only'
```

代码中保留了 `cop_mode='full'`，仅用于消融对照，不建议作为主方法。

---

## 7. 实验设置

### 7.1 评估口径

所有实验均使用 KITTI validation set，并采用 AP_R40 3D 指标。模型选择方式保持一致：

> 使用 `checkpoint_best.pth`，即按 `Car AP_R40 3D Moderate` 选择 best checkpoint。

这是为了与 MonoDETR 和 KITTI 3D 检测常用汇报口径保持一致。

### 7.2 对比实验

当前 P2 分析包含三组实验：

1. baseline：原始 MonoDETR 三次重复训练。
2. Full CoP replacement：替换 dim/angle/depth 全属性头，三次重复训练。
3. Depth-only CoP：只替换 depth head，三次重复训练。

所有实验保持相同数据类别设置：

```yaml
writelist: ['Car', 'Pedestrian', 'Cyclist']
```

---

## 8. 实验结果与分析

### 8.1 三次重复实验均值

| 方法 | Car Mod | Car Hard | Ped Mod | Ped Hard | Cyc Mod | Cyc Hard | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline mean±std | 19.3613±0.6280 | 16.3624±0.8570 | 5.4446±0.3652 | 4.3327±0.2982 | 4.4688±0.6479 | 4.1673±0.5800 | 原始三类基线 |
| Full CoP replacement mean±std | 18.9896±0.9216 | 15.9353±0.7208 | 6.2343±0.6636 | 4.8963±0.5455 | 4.0119±0.9861 | 3.6968±0.7612 | Ped 提升，但 Car/Cyc 下降 |
| Depth-only CoP mean±std | 19.8446±0.8441 | 16.3172±0.3251 | 6.6312±0.5641 | 5.0365±0.2055 | 4.6792±0.7122 | 4.1179±0.2878 | 当前 P2 最优方案 |

### 8.2 Depth-only CoP 相对 baseline 的提升

| 类别 | Moderate 差值 | Hard 差值 | 分析 |
|---|---:|---:|---|
| Car | +0.4833 | -0.0452 | 主指标提升，Hard 基本持平 |
| Pedestrian | +1.1866 | +0.7038 | 提升明显且稳定，是最强正向信号 |
| Cyclist | +0.2105 | -0.0495 | Moderate 小幅提升，Hard 基本持平 |

结果说明，Depth-only CoP 在不明显损伤 Car 和 Cyclist 的前提下，对 Pedestrian 带来显著提升。这符合本论文面向小目标 3D 检测的目标。

### 8.3 Depth-only CoP 相对 Full CoP 的提升

| 类别 | Moderate 差值 | Hard 差值 | 分析 |
|---|---:|---:|---|
| Car | +0.8550 | +0.3819 | 保留原 dim/angle 路径后，Car 稳定性恢复 |
| Pedestrian | +0.3969 | +0.1402 | 在 Full CoP 已提升 Ped 的基础上继续提升 |
| Cyclist | +0.6673 | +0.4210 | 明显优于 Full CoP replacement |

这组消融证明，Depth-only 不是为了简化而简化，而是更符合 MonoDETR 当前结构的稳定改法。

### 8.4 结论

Depth-only CoP 可以作为 P2 最终方案。它满足以下条件：

- 相比 baseline，Car Moderate 提升。
- Pedestrian Moderate 和 Hard 明显提升。
- Cyclist Moderate 小幅提升。
- Car Hard 和 Cyclist Hard 基本不伤。
- 相比 Full CoP replacement，所有关键指标更稳定。

因此，P2 最终不应表述为“完整 CoP 替换属性头”，而应表述为：

> 在保留 MonoDETR 原有属性预测稳定性的基础上，引入属性条件特征增强 per-query 深度预测。

---

## 9. 论文写作建议

### 9.1 方法章节可采用的表述

可直接转写为：

> MonoDETR 中的深度预测由 query-level 回归深度、几何深度以及密集深度图采样结果共同融合得到。其中 query-level 回归深度直接影响最终 3D 框深度。然而，原始 depth head 仅基于 decoder query feature 独立回归深度，未显式建模目标尺寸、朝向与深度之间的条件关系。针对这一问题，本文提出属性条件引导的深度预测头增强模块。该模块首先从 query feature 中提取尺寸相关特征，再结合尺寸特征构建朝向相关特征，最后将 query feature、尺寸特征和朝向特征共同用于深度回归。为了保持原始检测器中尺寸和朝向预测分支的稳定性，本文不直接替换最终的尺寸和朝向输出，仅将链式属性特征用于增强 per-query 深度预测。

### 9.2 消融章节可采用的表述

可直接转写为：

> 为验证链式属性建模的作用范围，本文比较了 Full CoP replacement 与 Depth-only CoP 两种实现方式。Full CoP replacement 同时替换尺寸、朝向和深度预测头，虽然提升了 Pedestrian 类别性能，但导致 Car 和 Cyclist 类别性能下降，说明直接替换全部属性预测头会破坏 MonoDETR 原有预测分支的稳定性。相比之下，Depth-only CoP 保留原始尺寸和朝向预测路径，仅增强深度预测头，在三次重复实验中取得更稳定的综合性能。因此，本文最终采用 Depth-only CoP 作为 P2 模块。

### 9.3 论文中的方法名称建议

中文名称建议：

- 属性条件引导的深度预测头增强模块
- 基于链式属性建模的深度预测增强模块
- 面向单目 3D 小目标检测的属性条件深度预测模块

英文名称建议：

- Attribute-conditioned Depth Prediction Head
- Chain-guided Depth Prediction Head
- Depth-only Chain-of-Prediction Head

推荐使用第一个中文名称：

> 属性条件引导的深度预测头增强模块

原因是该名称比 CoP 更准确，避免被认为是完整复现 MonoCoP，同时突出本文实际做法：只增强 depth head。

---

## 10. 与 P1 的关系

P1 和 P2 不是重复改进，而是作用于不同层级：

| 模块 | 改进层级 | 解决问题 | 作用对象 |
|---|---|---|---|
| P1 SCA-FPN | 特征提取 / neck | 小目标多尺度特征弱 | Transformer 输入前的多尺度特征 |
| P2 Depth-only CoP | 预测头 / depth head | per-query 深度预测不稳定 | Decoder query 的深度回归头 |

因此，论文总方法可以组织为双模块框架：

```text
输入图像
  -> Backbone
  -> P1: SCA-FPN 多尺度小目标特征增强
  -> Depth-aware Transformer
  -> P2: 属性条件引导的 Depth-only CoP 深度预测头
  -> 3D 检测结果
```

这种组织方式逻辑清晰：

- P1 解决“看不清小目标”的问题。
- P2 解决“看到了但深度估不准”的问题。

---

## 11. 当前 P2 的局限性

Depth-only CoP 虽然已表现出稳定正收益，但仍有局限：

1. Cyclist Hard 没有明显提升，只是基本持平。
2. 该方法主要增强 depth head，没有直接解决 query 分配和小目标召回问题。
3. 仍依赖 MonoDETR 原有三源深度平均融合，没有进一步优化融合策略。
4. 当前结论基于 KITTI validation set，未验证 test set 或其它数据集。

这些局限可以在论文中诚实说明，也可以引出未来工作：

- 引入多假设深度预测以处理更强的深度歧义。
- 设计小目标 query 分配或显著性监督机制。
- 改进三源深度融合的自适应性。

---

## 12. 最终结论

P2 的研究过程可以总结为：

1. 早期 Dynamic Bins 和几何深度加权融合没有稳定提升，因为它们主要间接影响最终深度。
2. 分析后确认 per-query `depth_embed` 是更直接的改进位置。
3. Full CoP replacement 证明属性链式建模对 Pedestrian 有价值，但全属性头替换会破坏模型稳定性。
4. Depth-only CoP 保留原始 dim/angle 分支，只增强 depth head，在三次重复实验中取得更稳定的综合提升。
5. 因此，P2 最终定稿为“属性条件引导的深度预测头增强模块”。

最终可用于论文的核心结论是：

> 通过在 per-query 深度回归头中引入尺寸和朝向相关的条件特征，Depth-only CoP 能够增强 MonoDETR 对单目深度歧义的建模能力。在三次重复实验中，该模块在保持 Car 和 Cyclist 性能基本稳定的同时，显著提升 Pedestrian 的 3D 检测性能，验证了属性条件深度预测对小目标单目 3D 检测的有效性。

---

## 17. 论文写作补充版：如何把 P2 写成一个完整创新点

本节用于后续直接拆分到论文的“方法设计”“实验分析”“消融实验”和“章节小结”中。前面章节更偏研究记录，本节更偏论文表达。

### 17.1 P2 的一句话定义

P2 可以定义为：

> 针对 MonoDETR 中 query-level 深度回归头缺少显式属性条件约束的问题，本文提出一种属性条件引导的 Depth-only CoP 深度预测头，在不改变原始尺寸、朝向、loss 和 decode 接口的前提下，将尺寸相关特征和朝向相关特征引入深度回归过程，从而增强单目 3D 小目标检测中的深度预测稳定性。

这句话里有几个关键点：

- “query-level 深度回归头”说明改进位置。
- “属性条件引导”说明技术思想。
- “Depth-only”说明没有替换全部属性头。
- “不改变 loss 和 decode 接口”说明工程风险低、消融清晰。
- “小目标深度预测稳定性”说明论文目标。

### 17.2 论文中 P2 应该回答的三个问题

写 P2 时要围绕三个问题展开：

1. 为什么要改 depth head？

原始 MonoDETR 虽然有 dense depth map、depth positional embedding 和三源深度融合，但最终 3D 框深度仍直接依赖 `depth_embed(hs)`。前期 Dynamic Bins 和三源加权融合实验没有稳定提升，说明间接调整深度图或融合权重不足以解决最终实例级深度回归问题。因此 P2 选择直接增强 per-query depth head。

2. 为什么引入尺寸和朝向条件？

单目 3D 检测中，深度、尺寸和朝向具有几何耦合关系。目标真实尺寸会影响 2D 投影尺度，目标朝向会影响可见形态和投影边界。对于行人和骑行者等小目标，图像纹理线索弱，深度估计更需要这种结构化属性条件。

3. 为什么只做 Depth-only，而不是 Full CoP？

Full CoP replacement 的实验说明，链式属性建模对 Pedestrian 有帮助，但直接替换 dim / angle / depth 全部预测头会破坏 MonoDETR 原有属性分支的稳定性，导致 Car 和 Cyclist 下降。Depth-only CoP 保留原始 dim / angle 输出，只把属性链作为 depth head 内部条件特征，因此收益更稳定。

### 17.3 可作为论文小节标题的版本

如果 P2 单独作为一个方法小节，可以用以下标题：

- 属性条件引导的深度预测头
- 基于链式属性条件的深度回归增强
- 面向小目标的 Depth-only 属性条件深度预测
- Attribute-conditioned Depth Prediction Head

推荐中文标题：

> 属性条件引导的深度预测头

这个标题比“CoP”更适合毕业论文，因为它直接描述你的实际改动，不会让读者误以为你完整复现了某个外部方法。

### 17.4 方法结构的论文表达

原始 MonoDETR 中，decoder query feature `h` 同时输入多个并行预测头：

```text
h -> class head
h -> box head
h -> angle head
h -> depth head
```

其中 depth head 与尺寸、朝向预测之间没有显式条件传播。P2 将 depth head 改为：

```text
h -> dim_feat
[h, dim_feat] -> angle_feat
[h, dim_feat, angle_feat] -> depth_feat
(depth_feat + dim_feat + angle_feat) -> depth_reg
```

但最终监督输出仍保持：

```text
pred_3d_dim = original dim branch
pred_angle  = original angle branch
pred_depth  = fused depth from CoP depth_reg, geo depth, map depth
```

这就是 Depth-only 的核心：属性链只服务于深度回归，不接管最终属性预测。

### 17.5 公式化表达

设 decoder 输出的 query feature 为 `h_i`，其中 `i` 表示第 `i` 个 object query。原始深度回归为：

```text
r_i = G_depth(h_i)
```

其中 `r_i = [r_i^d, r_i^u]`，分别表示深度 logit 和不确定性项。

P2 中，深度回归变为：

```text
f_i^s = F_s(h_i)
f_i^o = F_o([h_i, f_i^s])
f_i^z = F_z([h_i, f_i^s, f_i^o])
f_i^z = f_i^z + f_i^s + f_i^o
r_i = W_z f_i^z + b_z
```

其中：

- `f_i^s` 表示尺寸相关条件特征。
- `f_i^o` 表示朝向相关条件特征。
- `f_i^z` 表示深度回归特征。
- `r_i` 的维度仍为 2，接口与 baseline 一致。

随后深度 logit 转换为回归深度：

```text
D_i^reg = 1 / sigmoid(r_i^d) - 1
```

最终仍采用 MonoDETR 原始三源融合：

```text
D_i = (D_i^reg + D_i^geo + D_i^map) / 3
```

### 17.6 实验结论的稳妥表述

当前数据最稳的结论是：

> Depth-only CoP 对 Pedestrian 的提升最稳定，对 Car 和 Cyclist 基本不造成明显退化。

可以写：

> 在三次重复实验中，Depth-only CoP 将 Pedestrian Moderate / Hard 的 AP_R40 3D 从 5.4446 / 4.3327 提升到 6.6312 / 5.0365，分别提升 1.1866 和 0.7038。Car Moderate 小幅提升 0.4833，Car Hard 基本持平；Cyclist Moderate 小幅提升，Cyclist Hard 基本持平。该结果表明，属性条件深度预测能够改善小目标，尤其是行人类目标的深度估计质量。

不要写：

> 本方法显著提升所有类别。

因为 Cyclist Hard 和 Car Hard 只是基本持平，不支持这种表述。

### 17.7 消融实验的论文逻辑

消融实验建议按以下逻辑讲：

1. Baseline：原始 MonoDETR，属性头并行预测。
2. Full CoP Replacement：尺寸、朝向、深度全部链式替换。
3. Depth-only CoP：只增强深度头，保留原始尺寸和朝向预测路径。

消融结论：

> Full CoP Replacement 虽然提升 Pedestrian，但使 Car 和 Cyclist 下降，说明直接替换全部属性预测头会破坏原模型稳定性。Depth-only CoP 保留原始 dim / angle 分支，仅将属性条件特征用于深度预测，在 Pedestrian 上取得更稳定提升，并保持其他类别基本稳定。因此，最终采用 Depth-only 设计。

### 17.8 建议绘制的 P2 模块图

论文中建议画一张 P2 模块图，重点突出“保留原路径，只增强 depth head”：

```text
Decoder query feature h
        |
        |--------------------> Original dim branch ----> pred_3d_dim
        |
        |--------------------> Original angle branch --> pred_angle
        |
        |----> dim_feat
                |
                v
              concat(h, dim_feat) -> angle_feat
                                      |
                                      v
                  concat(h, dim_feat, angle_feat) -> depth_feat
                                                        |
                                                        v
                                                   depth_reg
                                                        |
                                                        v
                               three-source fusion with geo/map depth
                                                        |
                                                        v
                                                   pred_depth
```

图注可写：

> The proposed module only replaces the query-level depth regression head. The original dimension and orientation prediction branches are preserved for stability, while their corresponding conditional features are used to guide depth regression.

### 17.9 P2 与 P1 的组合叙事

后续如果 P1 + P2 组合实验有效，论文整体可以写成双模块框架：

- P1：在特征输入阶段增强小目标多尺度表达。
- P2：在预测输出阶段增强小目标深度回归。

可用叙事：

> P1 从特征层面提升小目标的可见性，P2 从预测头层面增强深度估计的几何约束。二者分别作用于检测流程的前端特征表达和后端深度回归，具有互补性。

### 17.10 最终论文贡献点写法

如果毕业论文需要列贡献点，P2 可以写成其中一条：

> 提出一种属性条件引导的 Depth-only 深度预测头。针对 MonoDETR 中 query-level 深度预测缺少显式属性约束的问题，该模块通过链式构造尺寸相关特征和朝向相关特征，引导深度回归，同时保留原始尺寸和朝向预测路径以维持训练稳定性。实验表明，该模块能够稳定提升 Pedestrian 类别的 3D 检测性能，并保持 Car 和 Cyclist 性能基本稳定。

### 17.11 后续实验记录要求

后续所有 P2 或 P1+P2 相关实验建议继续记录在 `experience_data.md`，至少包括：

- 实验名。
- 是否启用 P1。
- `use_cop` 和 `cop_mode`。
- best epoch。
- Car / Pedestrian / Cyclist 的 AP_R40 3D Easy / Moderate / Hard。
- 是否按 Car Moderate 选择 checkpoint。
- 与 baseline 或当前最好方法的差值。

这样后面写论文时不会再重新翻日志。
