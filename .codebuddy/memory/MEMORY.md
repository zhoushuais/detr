# 长期记忆

## 项目基本信息
- 研究生毕业论文项目：面向自动驾驶的单目3D小目标检测
- Baseline: MonoDETR (ICCV 2023), Car Mod AP3D R40 = 20.05
- KITTI数据集，三类：Car/Pedestrian/Cyclist
- 三个改进方向：P1 SCA-FPN (✅), P2 DBDU (负结果已收口), P3 SMCA (未开始)
- Git分支：main冻结在v0-baseline，各方向独立分支

## 实验评测规范
- 核心指标：AP_R40 3D
- 论文关键验证列：Pedestrian Hard, Cyclist Hard
- checkpoint_best按Car Mod选帧，Ped/Cyc顺带值噪声达±2 AP
- 判增益必须固定epoch(如195)且对齐baseline多seed区间

## P2教训
- Dynamic Bins和几何深度融合均负结果
- 根因：均只间接影响per-query depth_embed头，需直接强杠杆

## 安全网范式
- 零初始化权重/bias确保iter 0 = baseline
- 代码用cfg.get(key, default)读取，保证旧yaml可加载
- 注释tag：#[P1-SCA-FPN] / #[P2-DBDU] / #[P3-SMCA]

## 2026-06-16 论文调研结果
- CODEBUDDY.md已创建
- 完成2024-2026单目3D检测论文调研
- 第一梯队推荐：RARE(CVPR2026), MonoCoP(CVPR2026), Mono3DV(arXiv2026), Salience DETR(CVPR2024)
- 零成本第一步：启用DINO选项(use_dn=True等)
- 调研报告：research_report_mono3d_survey_2024_2026.md
