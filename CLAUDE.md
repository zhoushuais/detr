# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 仓库结构

由于此项目实际运行是在公司内部服务器，无法直接连接github，所以这个项目的改动都是先在本地上改动，然后我手动将修改后的代码复制到服务器上。

Git 仓库根目录是 `d:/zzl/MonoDETR/`，但 MonoDETR 源码树在它下一层的 [MonoDETR/](MonoDETR/) 里——所有训练、配置、代码相关命令都要从 `MonoDETR/` 进



## 配置驱动的消融

[configs/monodetr.yaml](MonoDETR/configs/monodetr.yaml) 是切换改进点的**唯一**位置。每个新功能都必须挂一个 config 开关，保证 baseline 可以原样复现。yaml 顶部 `model:` 里的 P1（SCA-FPN）块在注释里已经写好了规划中的四行消融——P2/P3 沿用相同模式。

新增 config 键时，在 `build()` 里用 `cfg.get('key', default)` 读取，这样旧的 yaml 文件（以及 `v0-baseline` tag）仍然能加载。

## 论文分支 + 注释约定

这是个**毕业论文项目**，规划了三个改进方向，每个方向都从 `v0-baseline` 单独开分支。要尊重的约定：

- `main` **永久冻结**在已复现的 baseline（tag `v0-baseline`，Car Mod AP3D R40 = 16.47）。不要把功能代码提交到 `main`。
- 每个改进方向**从 `v0-baseline` 派生**，不是从前一个方向派生，这样每个方向都可以单独消融：
  - `p1-sca-fpn` — Spatial-Channel Attention FPN + DCN lateral（**当前分支**）
  - `p2-dynamic-bins` — Dynamic Bins + Depth Uncertainty（DBDU）
  - `p3-smca` — Spatial-Modulated Co-Attention + 尺度感知 query
- 所有新代码必须用方向前缀标注注释，方便最后 `thesis-final` 合并时识别归属：
  - `# [P1-SCA-FPN]` 方向一
  - `# [P2-DBDU]` 方向二
  - `# [P3-SMCA]` 方向三

## 三个改进方向（论文核心框架）

三个方向覆盖 MonoDETR 的三个阶段——特征 → 深度 → 解码，分别打 baseline 的一个具体短板。每个方向都可单独消融，模块间留有自然依赖（P2 输出的 σ 可被 P3 复用为门控）。

### 方向一 P1：SCA-FPN + DCN lateral （✅ 已完成）

- **打的短板**：`input_proj` 只做 1×1 通道对齐，4 级特征 `{1/8, 1/16, 1/32, 1/64}` **没有任何跨尺度语义对齐**；同时 `depth_predictor` 内部的跨尺度融合是平凡求和 `(src_8 + src_16 + src_32) / 3`。远距离小目标在 1/32 已被压到 <2px，浅层有几何无语义，深层有语义无分辨率。
- **核心做法**（在 `input_proj` 与下游消费者之间原地改写 `srcs`）：
  1. **通道注意力 (CA)**：CBAM 风格，每层独立，AvgPool+MaxPool → 共享 MLP (reduction=16) → sigmoid。
  2. **局部空间注意力 (SA)**：每层 channel-wise avg+max → 7×7 Conv → sigmoid。
  3. **高层引导 (top-down guidance)**：只在最高层（1/64）算 SA，bilinear 上采到各低层，与该层的局部 SA 取均值。**"高语义老师指导低层学生"**，比 BiFPN/PaFPN 的对称融合更适合 3D 检测目标响应稀疏的特点。
  4. **DCN lateral**：lateral conv 从 `1×1 Conv + GN` 升级为 `3×3 DeformConv + GN`，offset 预测 conv 用 **zero-init** → iter 0 退化为标准 3×3，对训练统计量零扰动；offset 在训练中学习，让采样点贴上小/遮挡/截断目标的真实形状。
  5. **残差** `outs[i] = lateral(gated) + srcs[i]`：即使内部权重还没学好也不会让 baseline 倒退，关 `use_sca_fpn=False` 即回到原 MonoDETR。


### 方向二 P2：DBDU = Dynamic Bins + Depth Uncertainty + ADPM 残差精修

- **打的短板**：MonoDETR 的 `depth_predictor` 是固定 LID 80 桶 + 两个 3×3 Conv 的轻量结构，**深度桶在所有图像上共享**，远距离误差大，且前景深度图本身没有置信度信息（只有 per-query 的 σ）。
- **核心做法**：
  1. **Dynamic Bins**（DBDU-Depth, 于承峄）：每张图根据场景自适应分配深度桶边界，远距离桶变密、近距离桶变疏，对单调递减的距离分布更友好。
  2. **复用现有不确定性头**：`depth_embed = MLP(..., out=2)` 已经输出 `(depth_mean, log_sigma)`，[lib/losses/uncertainty_loss.py](MonoDETR/lib/losses/uncertainty_loss.py) 已在用——**P2 在这个 slot 之上叠加，不要新加并行机制**。
  3. **ADPM 残差精修**（杨帅兵）：在主深度预测后挂一个轻量残差头，专门修远距离/遮挡的系统性偏差。
  4. **3D 位置编码**对 z 做 log 压缩，缓解远距离深度梯度极小的问题。
  5. **几何深度融合**（苏卫星 MVPI / 贺宜）：基于 2D 框高度 + 相机焦距反推几何深度 `D_geo = f × H_real / h_2d`（用类别平均高度作 H_real），与回归深度通过 `depth_embed` 的 σ 做不确定性加权融合。免费的几何先验，且天然嵌进现有不确定性框架，**不引入新模块**。
  6. **新损失**：`L_ud`（不确定性下的深度回归损失）+ Bins 顺序约束损失（防止动态桶塌缩到非单调）。
- **代码位置**：
  - 主改 [lib/models/monodetr/depth_predictor/depth_predictor.py](MonoDETR/lib/models/monodetr/depth_predictor/depth_predictor.py)。
  - 新损失加在 [lib/losses/](MonoDETR/lib/losses/) 下，或扩展现有 `uncertainty_loss.py`。
- **Config 字段（规划）**：`use_dynamic_bins` / `num_dynamic_bins` / `use_adpm` / `use_geo_depth_fusion` / `depth_uncertainty_weight` / `bins_monotonic_weight`。
- **注释 tag**：`# [P2-DBDU]`。


### 方向三 P3：SMCA + 不确定性 V 门控 + 尺度感知 query


### thesis-final 整合阶段

三个方向消融全部跑完后开 `thesis-final` 分支按 P1→P2→P3 顺序合并。冲突预期集中在：

- [monodetr.py](MonoDETR/lib/models/monodetr/monodetr.py) 的 `__init__`：三个方向都改了构造参数，按 `# [P1] / # [P2] / # [P3]` 三段顺序拼接即可。
- [monodetr.py](MonoDETR/lib/models/monodetr/monodetr.py) 的 `forward`：插入顺序为 `input_proj` → SCA-FPN (P1) → depth_predictor (P2 改) → transformer (P3 改)。
- [monodetr.yaml](MonoDETR/configs/monodetr.yaml)：三个方向各自的 config 块按方向 tag 分块即可，无逻辑冲突。

## 评估：到底要看哪些数

KITTI 评估会同时打印 AP 和 AP_R40，在 IoU={Car 0.7, Ped/Cyc 0.5} 下分 Easy/Mod/Hard。对这篇论文来说真正重要的列是：

- **汇报指标**：`AP_R40 3D`（2019 年之后的标准是 40 点插值）。
- **官方排名列**：Moderate。
- **支撑"小目标"论文卖点的列**：Pedestrian AP_R40@0.5 3D **Hard** 和 Cyclist AP_R40@0.5 3D **Hard**——只有这两列能验证"小/远目标"的论文叙事。仅 Car Mod 涨点不够说明问题。

把评估结果贴回对话时，用户通常想第一时间看到 **Ped/Cyc Hard** 是否动了，而不只是 Car Mod。不要只看 Car Mod 涨了就过早庆祝（baseline 对照数以底部"实验数据"表为准）。

## 实验数据
存放每个改进方向对应git分支的实验数据，这里的实验数据我在每次训练结束后会手动加入。
P2 改进方向的实验数据表如下：

|  P2方向改进实验数据 |                              |            |                                     |      |         |          |         |
|:-------------------:|:----------------------------:|:----------:|:-----------------------------------:|:----:|:-------:|:--------:|:-------:|
|       实验名称      |           配置说明           |    日期    |                 类别                | 指标 |   Easy  | Moderate |   hard  |
|       baseline      |    之前的baseline训练数据    | 2026/06/08 |     Car AP_R40@0.70, 0.70, 0.70:    |  3d  | 27.4208 |  20.0496 | 16.8207 |
|                     |                              |            | Pedestrian AP_R40@0.50, 0.50, 0.50: |  3d  |  6.9518 |  5.3569  |  4.4932 |
|                     |                              |            |   Cyclist AP_R40@0.50, 0.50, 0.50:  |  3d  |  7.1343 |  3.5286  |  3.2470 |
|                     |                              |            |                                     |      |         |          |         |
|   P2-Dynamic Bins   |  第一次改进Dynamic Bins训练  | 2026/06/09 |     Car AP_R40@0.70, 0.70, 0.70:    |  3d  | 26.0083 |  19.2272 | 16.0419 |
|                     |                              |            | Pedestrian AP_R40@0.50, 0.50, 0.50: |  3d  |  7.5435 |   5.5830 |  4.4408 |
|                     |                              |            |   Cyclist AP_R40@0.50, 0.50, 0.50:  |  3d  |  8.2231 |  4.3122  |  4.0910 |
|                     |                              |            |                                     |      |         |          |         |
|   P2-Dynamic Bins   | 动态桶塌缩已修(LID 有界扰动) | 2026/06/10 |     Car AP_R40@0.70, 0.70, 0.70:    |  3d  | 26.6782 |  20.0877 | 16.2687 |
|                     |                              |            | Pedestrian AP_R40@0.50, 0.50, 0.50: |  3d  |  8.1280 |  5.9498  |  4.7418 |
|                     |                              |            |   Cyclist AP_R40@0.50, 0.50, 0.50:  |  3d  |  8.9777 |  4.1474  |  3.7591 |
|                     |                              |            |                                     |      |         |          |         |
| 几何深度 σ 加权融合 | clamp-linear 版              | 2026/06/12 |     Car AP_R40@0.70, 0.70, 0.70:    |  3d  | 26.3388 |  19.3180 | 16.1783 |
|                     |                              |            | Pedestrian AP_R40@0.50, 0.50, 0.50: |  3d  |  8.6121 |  6.9782  |  5.6236 |
|                     |                              |            |   Cyclist AP_R40@0.50, 0.50, 0.50:  |  3d  |  7.9408 |  3.8226  |  3.6795 |