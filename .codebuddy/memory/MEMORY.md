# 长期记忆

## 项目基本信息
- 研究生毕业论文项目：面向自动驾驶的单目3D小目标检测
- Baseline: MonoDETR (ICCV 2023), Car Mod AP3D R40 = 20.05
- KITTI数据集，三类：Car/Pedestrian/Cyclist
- 三个改进方向：P1 SCA-FPN (✅，独立分支p1-sca-fpn), P2 (本分支，重新规划), P3 (未开始)
- Git分支：main冻结在v0-baseline，当前在P2分支

## 实验评测规范
- 核心指标：AP_R40 3D
- 论文关键验证列：Pedestrian Hard, Cyclist Hard
- 论文采用标准做法：以checkpoint_best（按Car Mod AP3D R40选帧）的结果为最终汇报数据
- 内部注意事项：Ped/Cyc在checkpoint_best帧受选帧噪声影响（历史实验中单点漂移达±2 AP），消融判断时建议确认主结果经多seed验证

## 旧P2教训（p2-dynamic-bins分支）
- Dynamic Bins和几何深度融合均负结果
- 根因：均只间接影响per-query depth_embed头，需直接强杠杆
- 不要再碰bin边界调整和已有深度源的重新加权

## 代码规范
- 代码用cfg.get(key, default)读取，保证旧yaml可加载
- 注释tag：# [P1-SCA-FPN] / # [P2-KHYP]

## 2026-06-16
- P2改进方向重新规划：RARE-style K假设深度预测 + DINO基线增强
- P2_improvement_plan.md 包含完整方案
- P1 SCA-FPN 分析：方向合理，打的短板真实存在，工程实践健康
- 实施路线：先启用DINO baseline → 再实现K假设机制 → 备选MonoCoP链式预测
