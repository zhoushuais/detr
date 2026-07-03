# CODEBUDDY.md This file provides guidance to CodeBuddy when working with code in this repository.

## 项目概述

这是研究生毕业论文项目，基于 MonoDETR (ICCV 2023) 进行单目 3D 小目标检测改进。Baseline 已复现（Car Mod AP3D R40 = 20.05），规划三个改进方向。

代码在本地修改后手动复制到公司内部服务器执行训练，服务器端通过 Docker 容器 (`monodetrzzl`) 和 conda 环境 (`monodetr_v2`) 运行。

## 常用命令

**训练**: 确保已编译 deformable attention ops（`cd lib/models/monodetr/ops/ && bash make.sh`），然后设置显卡后训练：
```bash
export CUDA_VISIBLE_DEVICES=6
python tools/train_val.py --config configs/monodetr.yaml 2>&1 | tee logs/train_output.log
```
服务器上用 screen 保持会话：`screen -S train_bev` 创建，`screen -r train_bev` 恢复。训练会自动每 epoch 保存 checkpoint，并按 Car Mod 3D AP 选出 best checkpoint。

**仅评估**: `python tools/train_val.py --config configs/monodetr.yaml -e`（使用 checkpoint_best.pth）。修改 tester_helper.py 中的 checkpoint_path 可指定权重文件。

**可视化**: `python picture_tool/visualize.py --split val --vis_threshold 0.3`（需配置 data_root、result_dir、output_dir 参数）。

**数据集路径**: KITTI 放在 `../data/KITTIDataset/`（相对于仓库根目录），包含 ImageSets/、training/、testing/ 子目录。可在 yaml 中修改 `dataset/root_dir`。

## 代码架构

### 整体流程

MonoDETR 是首个基于 DETR 的单目 3D 检测器，无需额外深度监督、anchor 或 NMS。核心流程：ResNet50 backbone 提取多尺度特征 → DepthPredictor 预测密集深度图 → DepthAwareTransformer（encoder 用 MSDeformAttn 做多尺度特征交互，decoder 第一层用深度交叉注意力注入几何先验）→ 五个预测头（class、2D bbox+3D center、3D dim、angle、depth）→ 三源深度融合（回归深度 + 几何深度 + 深度图）→ 匈牙利匹配 + 多损失加权。

### 目录结构

- `lib/models/monodetr/monodetr.py` — 主模型文件。包含 `MonoDETR`（模型主体，构建 backbone、depth_predictor、transformer 和五个预测头）、`SetCriterion`（损失函数，含匈牙利匹配和 8 种损失）、`build()`（工厂函数组装所有组件）
- `lib/models/monodetr/depth_predictor/depth_predictor.py` — 深度预测器。多尺度特征（1/8、1/16、1/32）融合后经 depth_head → depth_classifier 输出逐像素深度分类，用 LID 分桶（或 [P2-DBDU] 动态分桶）
- `lib/models/monodetr/depth_predictor/ddn_loss/` — 深度图监督损失（DDNLoss），将 GT 稀疏深度转为密集深度图后做 focal loss
- `lib/models/monodetr/depthaware_transformer.py` — 深度感知 Transformer。VisualEncoder 用 MSDeformAttn 做多尺度可变形自注意力；DepthAwareDecoder 每层依次执行深度交叉注意力（query 与 depth_pos_embed 交互）、自注意力、可变形交叉注意力（query 与图像特征交互）、FFN，支持迭代 bbox refine
- `lib/models/monodetr/monodetr_head.py` — DAB-DETR/DINO 风格的 query 初始化和 bbox 嵌入
- `lib/models/monodetr/monodetr_transforms.py` — 训练/推理用目标构建、后处理和 DAB-DETR 转换
- `lib/models/monodetr/matcher.py` — 匈牙利匹配器，支持 class、bbox、giou、3dcenter 四种匹配代价
- `lib/models/monodetr/position_encoding.py` — 正弦/可学习位置编码
- `lib/models/monodetr/ops/` — 多尺度可变形注意力 CUDA 实现，首次使用需编译 `bash make.sh`
- `lib/losses/` — 独立损失模块：focal_loss、dim_aware_loss（维度感知 L1）、uncertainty_loss（拉普拉斯 aleatoric 不确定性）
- `lib/helpers/` — 训练管线：dataloader_helper（KITTI DataLoader）、model_helper、optimizer_helper（AdamW，bias 参数不 decay）、scheduler_helper（阶梯衰减，125/165 epoch 各 ×0.1）、trainer_helper（主训练循环，每 epoch 保存 checkpoint 并验证）、tester_helper（推理 + KITTI 官方评估）、decode_helper（从输出提取检测结果并解码为物理坐标）
- `lib/datasets/` — KITTI 数据集加载、预处理、标注解析
- `configs/monodetr.yaml` — 全局配置，所有改进点通过 config 开关控制
- `tools/train_val.py` — 训练/评估入口脚本

### 配置驱动的改进开关

`configs/monodetr.yaml` 是切换改进点的唯一位置。每个新功能必须挂 config 开关，在 `build()` 中用 `cfg.get('key', default)` 读取，保证旧 yaml 可加载。当前规划了三个方向（`p1-sca-fpn`、`p2-dynamic-bins`、`p3-smca`），各自从 `v0-baseline` tag 派生独立分支，不在 main 上开发。

### 三大改进方向

**P1 SCA-FPN（已完成）**: 在 input_proj 中原地替换 srcs，加入通道注意力、局部空间注意力、高层语义引导（1/64 层 SA 上采样指导低层）、DCN lateral（3×3 DeformConv zero-init 安全起步）、残差连接。打的是多尺度特征间没有跨尺度语义对齐的短板。

**P2 DBDU（当前分支，两个子机制均负结果已收口）**: 尝试了 Dynamic Bins（每图自适应深度分桶）和几何深度 σ 加权融合（三源逆方差融合替代 /3 平均），均对 AP 无增益。根因已坐实：两者都只间接影响最终 3D 框深度——bin 中心只经 weighted_depth 影响深度位置编码和前景深度图 aux loss，融合权重只重新加权已有源——而最终深度由 per-query depth_embed 头主导。要涨点必须上直接作用于 per-query 深度的强杠杆组件。

**P3 SMCA（未开始）**: Spatial-Modulated Co-Attention + 尺度感知 query。

### 注释约定

所有新代码必须用方向前缀标注：`# [P1-SCA-FPN]`、`# [P2-DBDU]`、`# [P3-SMCA]`，方便 thesis-final 合并时识别归属。

### 评估指标

核心指标是 AP_R40 3D（40 点插值）。论文关注 KITTI Moderate 列（官方排名列），但验证"小目标"论文叙事必须看 Pedestrian Hard 和 Cyclist Hard——这两列才是真正反映小/远目标检测提升的关键。不要只看 Car Mod 涨点就下结论。

评测方法论要点：checkpoint_best 按 Car Mod 选帧，Ped/Cyc 报的是 argmax-Car 帧的顺带值，单次 run 的 Ped Hard 噪声实测达 ±2 AP。判增益必须固定 epoch（如 epoch195）且对齐 baseline 多 seed 区间，禁止用 checkpoint_best 的单点数据下结论。

### 零初始化安全网模式

本项目反复使用的安全网范式：新模块用零初始化权重/bias，确保 iter 0 时精确退化为 baseline；仅当优化器主动调大参数（证明改进确实降 loss）时才获得权重。这保证了下限 = baseline，不会因新模块初始化噪声导致训练崩溃（如几何融合首版无门时整个检测器崩溃到 Car Mod 10.9）。

### Git 分支策略

- `main` 永久冻结在 `v0-baseline` tag（Car Mod AP3D R40 = 20.05）
- 三个改进方向各自从 `v0-baseline` 派生独立分支，保证各自可单独消融
- 最终 `thesis-final` 分支按 P1→P2→P3 顺序合并
- 合并冲突集中在 `monodetr.py` 的 `__init__`/`forward` 和 `monodetr.yaml`，按方向 tag 分块拼接即可
