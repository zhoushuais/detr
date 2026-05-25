# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 仓库结构

由于此项目实际运行是在公司内部服务器，无法直接连接github，所以这个项目的改动都是先在本地上改动，然后我手动将修改后的代码复制到服务器上。

Git 仓库根目录是 `d:/zzl/MonoDETR/`，但 MonoDETR 源码树在它下一层的 [MonoDETR/](MonoDETR/) 里——所有训练、配置、代码相关命令都要从 `MonoDETR/` 进，**不是**从仓库根目录。

```
d:/zzl/MonoDETR/                 <- git 仓库根，放 mingling.txt 和参考论文
├── MonoDETR/                    <- 所有开发命令的工作目录
│   ├── GIT_WORKFLOW.md          <- 三方向分支策略 + 消融实验流程
│   ├── P1_SCA_FPN_Summary.md    <- 方向一改进方案文档
│   ├── README.md                <- 上游 MonoDETR README（环境安装步骤）
│   ├── configs/monodetr.yaml    <- 唯一的超参 + 消融开关来源
│   ├── lib/                     <- 模型、损失、数据集、辅助代码
│   ├── tools/train_val.py       <- 单一入口（训练或 `-e` 仅评估）
│   ├── train.sh / test.sh       <- train_val.py 的一行封装
│   └── outputs/<model_name>/    <- 权重 + 训练日志（已 gitignore）
├── mingling.txt                 <- 用户的运维速记（Docker/screen/GPU），不是文档
└── 相关论文/                     <- 参考论文 PDF（不是代码）
```


## 架构（动代码前必须先理解的部分）

MonoDETR 是深度引导的 Deformable-DETR 单目 3D 检测器。模型在 [`build()` (monodetr.py:583)](MonoDETR/lib/models/monodetr/monodetr.py#L583) 中装配，前向路径在 [`MonoDETR.forward` (monodetr.py:177)](MonoDETR/lib/models/monodetr/monodetr.py#L177)：

```
images
  └─ backbone (ResNet-50, 多尺度输出 stride 8/16/32)
       └─ input_proj  (1x1 conv → hidden_dim=256, 再用 3x3-stride-2 下采到 1/64)
            └─ [P1-SCA-FPN] sca_fpn (空间-通道注意力 FPN，对 srcs 原地操作)
                 ├─ depth_predictor          (前景深度图, LID 离散化 80 bins)
                 └─ depthaware_transformer   (encoder + 带深度交叉注意力的 decoder, 50 个 query)
                      └─ 预测头: class / 2D box / 3D dim / angle / depth (mean+log-σ)
```

改代码前要先吃透这三件事：

1. **`srcs` 是共享的多尺度特征列表**。`input_proj` 输出 `[1/8, 1/16, 1/32, 1/64]` 四级特征，全部都是 `hidden_dim=256`；`depth_predictor` 和 `depthaware_transformer` 消费的是**同一个** list。所以任何特征层面的改进（P1）只需要在这两个消费者之间修改 `srcs` 即可——SCA-FPN 的插入位置就在 [monodetr.py:210-211](MonoDETR/lib/models/monodetr/monodetr.py#L210)。

2. **深度既是特征也是输出**。`depth_predictor` 一方面输出前景深度图 logits 用于辅助损失，一方面输出深度位置编码喂进 decoder。所以深度分支不能当作旁路任务——P2（动态 Bins + 不确定性）的改动会顺着这条链路流进 decoder 的 KV。

3. **per-query 深度已经有显式不确定性头**。`self.depth_embed = MLP(..., out=2, ...)` 输出 `(depth_mean, log_sigma)`，[lib/losses/uncertainty_loss.py](MonoDETR/lib/losses/uncertainty_loss.py) 已经在用了。P2（DBDU）方向是在这个 slot 之上叠加，**不要**再加一套并行的不确定性机制。

[MonoDETR/lib/models/monodetr/ops/](MonoDETR/lib/models/monodetr/ops/) 下的 Deformable Attention CUDA 算子是 vendored 的 Deformable-DETR 编译产物，按第三方代码对待。

## 配置驱动的消融

[configs/monodetr.yaml](MonoDETR/configs/monodetr.yaml) 是切换改进点的**唯一**位置。每个新功能都必须挂一个 config 开关，保证 baseline 可以原样复现。yaml 顶部 `model:` 里的 P1（SCA-FPN）块在注释里已经写好了规划中的四行消融——P2/P3 沿用相同模式。

新增 config 键时，在 `build()` 里用 `cfg.get('key', default)` 读取，这样旧的 yaml 文件（以及 `v0-baseline` tag）仍然能加载。

## 论文分支 + 注释约定

这是个**毕业论文项目**，规划了三个改进方向，每个方向都从 `v0-baseline` 单独开分支。完整的策略和 Git 流程在 [MonoDETR/GIT_WORKFLOW.md](MonoDETR/GIT_WORKFLOW.md) 和 [MonoDETR/P1_SCA_FPN_Summary.md](MonoDETR/P1_SCA_FPN_Summary.md) 里。要尊重的约定：

- `main` **永久冻结**在已复现的 baseline（tag `v0-baseline`，Car Mod AP3D R40 = 16.47）。不要把功能代码提交到 `main`。
- 每个改进方向**从 `v0-baseline` 派生**，不是从前一个方向派生，这样每个方向都可以单独消融：
  - `p1-sca-fpn` — Spatial-Channel Attention FPN + DCN lateral（**当前分支**）
  - `p2-dynamic-bins` — Dynamic Bins + Depth Uncertainty（DBDU）
  - `p3-smca` — Spatial-Modulated Co-Attention + 尺度感知 query
- 所有新代码必须用方向前缀标注注释，方便最后 `thesis-final` 合并时识别归属：
  - `# [P1-SCA-FPN]` 方向一
  - `# [P2-DBDU]` 方向二
  - `# [P3-SMCA]` 方向三
- 合并冲突会集中在 [monodetr.py](MonoDETR/lib/models/monodetr/monodetr.py)（`__init__` 和 `forward`）和 [configs/monodetr.yaml](MonoDETR/configs/monodetr.yaml)。把新 yaml 字段和新构造参数都按**带方向 tag 的代码块**组织是控制冲突的关键。

## 三个改进方向（论文核心框架）

三个方向覆盖 MonoDETR 的三个阶段——特征 → 深度 → 解码，分别打 baseline 的一个具体短板。每个方向都可单独消融，模块间留有自然依赖（P2 输出的 σ 可被 P3 复用为门控）。详细方案分别写在 [MonoDETR/P1_SCA_FPN_Summary.md](MonoDETR/P1_SCA_FPN_Summary.md)（P1 已完成）以及未来的 `P2_DBDU_Summary.md` / `P3_SMCA_Summary.md`。

### 方向一 P1：SCA-FPN + DCN lateral （✅ 已完成）

- **打的短板**：`input_proj` 只做 1×1 通道对齐，4 级特征 `{1/8, 1/16, 1/32, 1/64}` **没有任何跨尺度语义对齐**；同时 `depth_predictor` 内部的跨尺度融合是平凡求和 `(src_8 + src_16 + src_32) / 3`。远距离小目标在 1/32 已被压到 <2px，浅层有几何无语义，深层有语义无分辨率。
- **核心做法**（在 `input_proj` 与下游消费者之间原地改写 `srcs`）：
  1. **通道注意力 (CA)**：CBAM 风格，每层独立，AvgPool+MaxPool → 共享 MLP (reduction=16) → sigmoid。
  2. **局部空间注意力 (SA)**：每层 channel-wise avg+max → 7×7 Conv → sigmoid。
  3. **高层引导 (top-down guidance)**：只在最高层（1/64）算 SA，bilinear 上采到各低层，与该层的局部 SA 取均值。**"高语义老师指导低层学生"**，比 BiFPN/PaFPN 的对称融合更适合 3D 检测目标响应稀疏的特点。
  4. **DCN lateral**：lateral conv 从 `1×1 Conv + GN` 升级为 `3×3 DeformConv + GN`，offset 预测 conv 用 **zero-init** → iter 0 退化为标准 3×3，对训练统计量零扰动；offset 在训练中学习，让采样点贴上小/遮挡/截断目标的真实形状。
  5. **残差** `outs[i] = lateral(gated) + srcs[i]`：即使内部权重还没学好也不会让 baseline 倒退，关 `use_sca_fpn=False` 即回到原 MonoDETR。
- **代码位置**：
  - 新模块 [lib/models/monodetr/sca_fpn.py](MonoDETR/lib/models/monodetr/sca_fpn.py)（`ChannelAttention` / `SpatialAttention` / `DeformLateralConv` / `SCAFPN`）。
  - [monodetr.py:117-132](MonoDETR/lib/models/monodetr/monodetr.py#L117) 构造，[monodetr.py:210-211](MonoDETR/lib/models/monodetr/monodetr.py#L210) 在 forward 中插入到 `input_proj` 之后。
- **Config 开关**（[configs/monodetr.yaml](MonoDETR/configs/monodetr.yaml) `model:` 块）：`use_sca_fpn` / `sca_fpn_reduction` / `sca_fpn_kernel` / `sca_fpn_high_guidance` / `use_dcn_lateral` / `dcn_kernel`。
- **消融四行**（注释里已写好规划，**四行全部跑完**，详见底部"实验数据"表）：
  - r1 — `use_sca_fpn=False` → Baseline（05/18）
  - r2 — SCA-FPN, `high_guidance=False, dcn=False` → +CA/SA only（05/22 的 `use_high_guidance=False`，当时 DCN 也还没加入）
  - r3 — SCA-FPN, `high_guidance=True, dcn=False` → +top-down guide（05/20 的 `epoch195`，当时 DCN 还没加；05/21 的 `epoch230` 是同 config 长训对比，已验证训过 195 ep 会过拟合退化）
  - r4 — SCA-FPN, `high_guidance=True, dcn=True` → **Full P1（最终行）**（05/25）
- **实测消融阶梯**（Car Mod / Ped Mod / Cyc Mod，AP_R40 3D）：
  - r1 baseline: 18.72 / 5.20 / 4.35
  - r2 +CA/SA only: **18.59** / 6.36 / 4.73 → Car 几乎不动 (-0.13)，Ped/Cyc 温和涨（+1.16 / +0.38）。说明**局部双重注意力主要受益于小目标**。
  - r3 +top-down guide: **17.77** / 6.49 / 5.01 → **Car 反而退 (-0.95)**，Ped/Cyc 继续微涨。说明**高层 SA 反向调制低层会扰乱 Car 类的标准采样网格**——这是个反直觉但可解释的现象。
  - r4 Full (+DCN): **20.02** / 8.26 / 5.19 → Car 大涨（vs r1 +1.30, **vs r3 +2.25**），Ped 大涨（+3.06），Cyc 微涨（+0.84）。**DCN 一肩挑两个作用**——补偿 high_guidance 对 Car 采样网格的扰乱（zero-init offset 从恒等映射平滑学起），同时让 lateral 采样点在远距离小目标上自适应贴合。
- **方向一收尾状态**：✅ 消融完整 + 主指标涨 + 小目标显著涨。剩余仅需补充：(i) 按距离（0–20m / 20–40m / >40m）分段重算 AP 以强化小目标叙事；(ii) Params/FLOPs 对照表证明几乎零代价。这两项不需要重训，仅在现有 checkpoint 上做后处理。
- **借鉴文献**：CBAM (Woo+ECCV2018) / FPN (Lin+CVPR2017) / 杨帅兵 SCA-FPN（2026 计算机工程与应用，本方案直接来源）/ HS-FPN (Chen+CBM2024) / 李铖硕硕士论文（2024，多尺度对 MonoDETR 的必要性印证）。

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
- **借鉴文献**：于承峄 MonoDBDU（动态 Bins + 不确定性融合）/ GUPNet（深度不确定性）/ 杨帅兵 ADPM 残差精修 / 苏卫星 MVPI（几何深度与回归深度的不确定性加权融合）/ 贺宜（前景深度图 + 几何深度双分支）/ 李铖硕 ACmix。

### 方向三 P3：SMCA + 不确定性 V 门控 + 尺度感知 query

- **打的短板**：MonoDETR decoder 用的 Deformable Attention 每 query 只采 **8 个点**，对小目标采样不足；query 没有空间先验/尺度先验；视觉 cross-attn 单尺度 KV。
- **核心做法**：
  1. **SMCA（Spatial-Modulated Co-Attention）**（李铖硕 SC/DSC-MonoDETR）：在 decoder 视觉 cross-attn 上挂一个高斯权重图 `log G` 做加性调制，把注意力软约束到 query 的参考点附近。**关键陷阱**：高斯方差必须**可学习**，否则 Hard 难度会退化（原 SC-MonoDETR 论文实测）。
  2. **不确定性 V 门控**：把 P2 输出的 σ 作为 attention V 路径的门控，对深度不确定的 query 衰减其 V 贡献——这是 P2 → P3 的天然依赖。
  3. **尺度感知 query**：query embedding 附加尺度维度，让不同 query 专注于不同目标尺寸（结合 MonoATT 的自适应 token 思想）。
  4. **3D 框尺寸尺度监督**（王鑫威 SSQM）：把 GT 3D 框的 `(h, w, l)` 投影回 2D 期望尺度，作为 query 尺度头的辅助监督信号——比纯靠 3D 主损失反传给 query 尺度学得快得多，对小目标尤其有效。
  5. （备选）DU-Transformer 的 SW-MSA 替换某一层 self-attn。
- **代码位置**：主改 [lib/models/monodetr/depthaware_transformer.py](MonoDETR/lib/models/monodetr/depthaware_transformer.py) 的 decoder 视觉 cross-attn。
- **Config 字段（规划）**：`use_smca` / `smca_learnable_sigma` / `use_uncert_v_gate` / `use_scale_aware_query` / `query_scale_dim` / `use_3dsize_scale_sup` / `scale_sup_weight`。
- **注释 tag**：`# [P3-SMCA]`。
- **借鉴文献**：李铖硕 DSC-MonoDETR（高斯调制 co-attention）/ 王鑫威 SSQM（尺度监督 query 优化，3D 框尺寸作监督信号）/ 于承峄 DU-Transformer SW-MSA / Deformable DETR (Zhu+ICLR2021) / MonoATT（自适应 token）。
- **独立消融 fallback**：P3 单独消融时 P2 的 σ 不可用，用 `depth_embed` 自身的 log_sigma 作 V 门控替代。

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
这里的实验数据我在每次训练结束后会手动加入。实验数据表如下：

|                  实验名称                  |                            配置说明                           |    日期    |                 类别                | 指标 |   Easy  | Moderate |   hard  |
|:------------------------------------------:|:-------------------------------------------------------------:|:----------:|:-----------------------------------:|:----:|:-------:|:--------:|:-------:|
|                  baseline                  |                           源码未改动                          | 2026/05/18 |     Car AP_R40@0.70, 0.70, 0.70:    |  3d  | 25.0890 |  18.7150 | 15.6665 |
|                                            |                                                               |            | Pedestrian AP_R40@0.50, 0.50, 0.50: |  3d  |  6.7946 |  5.1998  |  4.1021 |
|                                            |                                                               |            |   Cyclist AP_R40@0.50, 0.50, 0.50:  |  3d  |  8.5283 |  4.3472  |  4.2742 |
|                                            |                                                               |            |                                     |      |         |          |         |
|             P1—SCA-FPN-epoch195            |            增加了方向一改进，SCA-FPN和high_guidance           | 2026/05/20 |     Car AP_R40@0.70, 0.70, 0.70:    |  3d  | 24.4012 |  17.7699 | 14.9842 |
|                                            |                                                               |            | Pedestrian AP_R40@0.50, 0.50, 0.50: |  3d  |  9.4759 |  6.4864  |  5.1922 |
|                                            |                                                               |            |   Cyclist AP_R40@0.50, 0.50, 0.50:  |  3d  | 10.0604 |  5.0128  |  4.5799 |
|                                            |                                                               |            |                                     |      |         |          |         |
|             P1—SCA-FPN-epoch230            |      在实验P1—SCA-FPN-epoch195的基础上只修改了epoch为230      | 2026/05/21 |     Car AP_R40@0.70, 0.70, 0.70:    |  3d  | 23.5432 |  16.7749 | 13.8605 |
|                                            |                                                               |            | Pedestrian AP_R40@0.50, 0.50, 0.50: |  3d  |  8.7971 |  6.4220  |  5.1153 |
|                                            |                                                               |            |   Cyclist AP_R40@0.50, 0.50, 0.50:  |  3d  |  7.3007 |  3.6262  |  3.5643 |
|                                            |                                                               |            |                                     |      |         |          |         |
|     P1—SCA-FPN-use_high_guidance=False     | 在P1—SCA-FPN-epoch195实验的基础上将use_high_guidance改为False | 2026/05/22 |     Car AP_R40@0.70, 0.70, 0.70:    |  3d  | 26.1064 |  18.5943 | 15.4058 |
|                                            |                                                               |            | Pedestrian AP_R40@0.50, 0.50, 0.50: |  3d  |  8.4195 |  6.3624  |  5.1062 |
|                                            |                                                               |            |   Cyclist AP_R40@0.50, 0.50, 0.50:  |  3d  |  9.8578 |  4.7323  |  4.4051 |
|                                            |                                                               |            |                                     |      |         |          |         |
| P1—SCA-FPN-use_high_guidance=True-DCN=True |         在P1—SCA-FPN-epoch195实验的基础上加了DCN的改进        | 2026/05/25 |     Car AP_R40@0.70, 0.70, 0.70:    |  3d  | 27.5843 |  20.0177 | 16.0858 |
|                                            |                                                               |            | Pedestrian AP_R40@0.50, 0.50, 0.50: |  3d  | 10.8044 |  8.2620  |  6.6623 |
|                                            |                                                               |            |   Cyclist AP_R40@0.50, 0.50, 0.50:  |  3d  |  9.8809 |  5.1913  |  4.4418 |
