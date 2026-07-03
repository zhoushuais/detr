# Research Plan: Monocular 3D Object Detection Improvements (2024-2026)

## 目标
调研 2024 年至 2026 年的单目 3D 目标检测论文，找出可以在 MonoDETR baseline 上尝试的改进方法，以提升 KITTI 上小/远目标（Pedestrian Hard, Cyclist Hard）的 AP。

## 背景
- Baseline: MonoDETR (ICCV 2023)，基于 DETR 的单目 3D 检测器
- P1 方向（SCA-FPN 特征增强）已完成
- P2 方向（Dynamic Bins + 几何深度融合）均为负结果，根因：间接影响最终深度，需直接作用于 per-query 深度的强杠杆
- P3 方向（SMCA）尚未开始，原规划是空间调制共注意力 + 尺度感知 query
- 关键瓶颈：远处小目标（深度误差大、特征分辨率低、遮挡严重）

## 调研方向

### 方向A：深度估计改进（直接作用于深度预测头）
搜索关键词：monocular depth estimation uncertainty, per-query depth refinement, depth auxiliary task, depth-aware loss
重点关注能直接改进 per-query 深度预测的方法（非间接的 bin 边界调整或源融合权重调整）

### 方向B：Transformer 架构改进（编码器/解码器/注意力）
搜索关键词：DETR 3D detection, deformable attention improvements, query design, decoder architecture, position encoding for 3D
重点关注可用于 DETR 风格检测器的 transformer 改进

### 方向C：小目标检测 / 多尺度特征增强
搜索关键词：small object 3D detection, multi-scale feature fusion, FPN improvements, high-resolution features for distant objects
重点关注远程小目标检测的针对性方法

### 方向D：训练策略 / 损失函数 / 数据增强
搜索关键词：monocular 3D detection loss function, depth supervision, auxiliary task, data augmentation for 3D
重点关注新颖的训练策略和损失

### 方向E：知识蒸馏 / 多模态融合 / 时序
搜索关键词：knowledge distillation monocular 3D, lidar supervision, temporal fusion monocular detection, stereo guidance
重点关注利用额外监督信号的方法

## 搜索策略

1. **Web Search**：ArXiv、Google Scholar、Papers With Code 等
   - 时间过滤：2024-2026
   - 会议重点：CVPR 2024/2025/2026、ICCV 2025、ECCV 2024、NeurIPS 2024/2025、ICLR 2024/2025/2026
   - KITTI benchmark 最新排名

2. **WeChat Article Search**：微信公众号高质量中文技术文章
   - 加载 wechat-article-search skill
   - 搜索关键词：单目3D检测、MonoDETR改进、小目标3D检测、深度估计不确定性

## 子代理分工

### Subagent 1: 深度估计与深度头改进 (Depth Estimation Focus)
搜索 2024-2026 年关于单目 3D 检测中深度估计改进的论文：
- 深度不确定性建模的新方法
- Per-pixel/per-query 深度优化策略
- 深度辅助任务设计
- 深度图质量提升方法

### Subagent 2: Transformer 架构与 Query 设计 (Architecture Focus)
搜索 2024-2026 年 DETR 系列在 3D 检测上的架构改进：
- 解码器注意力机制改进
- Query 初始化和优化策略
- 位置编码改进
- Encoder 特征增强

### Subagent 3: 小目标检测与特征增强 (Small Object & Feature Focus)
搜索 2024-2026 年针对小目标/远距离目标检测的方法：
- 多尺度特征融合新方法
- 高分辨率特征保留策略
- FPN 变体改进
- 针对 KITTI Pedestrian/Cyclist 的专门方法

### Subagent 4: 训练策略与损失函数 (Training & Loss Focus)
搜索 2024-2026 年训练策略和损失函数的新进展：
- 新损失函数设计
- 数据增强策略
- 知识蒸馏/多模态监督
- 训练技巧

## 信息整合方式

各子代理返回结果后，按以下维度整合：
1. 方法分类（直接深度改进 / 架构改进 / 特征增强 / 训练策略）
2. 对 MonoDETR 的适用性评估（能否直接嵌入？改动量多大？）
3. 预期增益方向（Car Mod / Ped Hard / Cyc Hard）
4. 优先级排序（高：直接作用于 per-query 深度的强杠杆；中：架构/python层改进；低：间接或改动量过大）
5. 论文信息汇总（标题、会议/期刊、年份、核心方法、KITTI 结果）
