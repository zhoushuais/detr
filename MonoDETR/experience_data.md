| P2方向改进实验数据-CoP |                        |                |                                     |      |         |          |         |
|:----------------------:|:----------------------:|:--------------:|:-----------------------------------:|:----:|:-------:|:--------:|:-------:|
|        实验名称        |        配置说明        |      日期      |                 类别                | 指标 |   Easy  | Moderate |   hard  |
|        baseline        |  源码未改动第一次训练  |   2026/05/18   |     Car AP_R40@0.70, 0.70, 0.70:    |  3d  | 25.0890 |  18.7150 | 15.6665 |
|                        |                        |                | Pedestrian AP_R40@0.50, 0.50, 0.50: |  3d  |  6.7946 |  5.1998  |  4.1021 |
|                        |                        |                |   Cyclist AP_R40@0.50, 0.50, 0.50:  |  3d  |  8.5283 |  4.3472  |  4.2742 |
|                        |                        |                |                                     |      |         |          |         |
|        baseline        |  源码未改动第二次训练  |   2026/06/25   |     Car AP_R40@0.70, 0.70, 0.70:    |  3d  | 28.0994 |  19.9693 | 17.3196 |
|                        |                        | best_epoch=142 | Pedestrian AP_R40@0.50, 0.50, 0.50: |  3d  |  7.1964 |  5.2696  |  4.2265 |
|                        |                        |                |   Cyclist AP_R40@0.50, 0.50, 0.50:  |  3d  |  7.6784 |  3.8903  |  3.5413 |
|                        |                        |                |                                     |      |         |          |         |
|        baseline        |  源码未改动第三次训练  |   2026/06/25   |     Car AP_R40@0.70, 0.70, 0.70:    |  3d  | 26.8978 |  19.3995 | 16.1012 |
|                        |                        | best_epoch=172 | Pedestrian AP_R40@0.50, 0.50, 0.50: |  3d  |  8.2200 |  5.8644  |  4.6695 |
|                        |                        |                |   Cyclist AP_R40@0.50, 0.50, 0.50:  |  3d  |  9.7978 |  5.1688  |  4.6865 |
|                        |                        |                |                                     |      |         |          |         |
|        CoP-run1        |   第一次CoP的改进训练  |   2026/06/23   |     Car AP_R40@0.70, 0.70, 0.70:    |  3d  | 26.4068 |  20.0078 | 16.7595 |
|                        |                        | best_epoch=139 | Pedestrian AP_R40@0.50, 0.50, 0.50: |  3d  |  7.2642 |  5.5270  |  4.3005 |
|                        |                        |                |   Cyclist AP_R40@0.50, 0.50, 0.50:  |  3d  | 10.1961 |  5.0253  |  4.4634 |
|                        |                        |                |                                     |      |         |          |         |
|        CoP-run2        |   第二次CoP的改进训练  |   2026/06/23   |     Car AP_R40@0.70, 0.70, 0.70:    |  3d  | 25.1427 |  18.2126 | 15.4227 |
|                        |                        | best_epoch=146 | Pedestrian AP_R40@0.50, 0.50, 0.50: |  3d  |  8.6663 |  6.3329  |  5.0172 |
|                        |                        |                |   Cyclist AP_R40@0.50, 0.50, 0.50:  |  3d  |  8.3416 |  3.9550  |  3.6860 |
|                        |                        |                |                                     |      |         |          |         |
|        CoP-run3        |   第三次CoP的改进训练  |   2026/06/24   |     Car AP_R40@0.70, 0.70, 0.70:    |  3d  | 24.2547 |  18.7485 | 15.6238 |
|                        |                        | best_epoch=152 | Pedestrian AP_R40@0.50, 0.50, 0.50: |  3d  |  9.2328 |  6.8431  |  5.3712 |
|                        |                        |                |   Cyclist AP_R40@0.50, 0.50, 0.50:  |  3d  |  6.9938 |  3.0555  |  2.9411 |
|                        |                        |                |                                     |      |         |          |         |
|   CoP-depth_only-run1  | cop_mode: 'depth_only' |   2026/06/26   |     Car AP_R40@0.70, 0.70, 0.70:    |  3d  | 27.2648 |  19.4775 | 16.1973 |
|                        |                        | best_epoch=143 | Pedestrian AP_R40@0.50, 0.50, 0.50: |  3d  |  8.7339 |  6.4438  |  5.1207 |
|                        |                        |                |   Cyclist AP_R40@0.50, 0.50, 0.50:  |  3d  |  9.0970 |  5.1612  |  4.0974 |
|                        |                        |                |                                     |      |         |          |         |
|   CoP-depth_only-run2  | cop_mode: 'depth_only' |   2026/06/26   |     Car AP_R40@0.70, 0.70, 0.70:    |  3d  | 27.9262 |  20.8101 | 16.6852 |
|                        |                        | best_epoch=154 | Pedestrian AP_R40@0.50, 0.50, 0.50: |  3d  |  9.3454 |  7.2651  |  5.1865 |
|                        |                        |                |   Cyclist AP_R40@0.50, 0.50, 0.50:  |  3d  |  7.3852 |  3.8612  |  3.8408 |
|                        |                        |                |                                     |      |         |          |         |
|   CoP-depth_only-run3  | cop_mode: 'depth_only' |   2026/06/27   |     Car AP_R40@0.70, 0.70, 0.70:    |  3d  | 26.6179 |  19.2462 | 16.0691 |
|                        |                        | best_epoch=159 | Pedestrian AP_R40@0.50, 0.50, 0.50: |  3d  |  8.2161 |  6.1847  |  4.8023 |
|                        |                        |                |   Cyclist AP_R40@0.50, 0.50, 0.50:  |  3d  | 10.7089 |  5.0153  |  4.4154 |
## 2026-06-25 阶段性结论记录

评估口径：所有结果均采用 `checkpoint_best.pth`，即按 `Car AP_R40 3D Moderate` 选 best checkpoint。当前三次 baseline 与三次 Full CoP replacement 的统计如下。

| 方法 | Car Mod | Car Hard | Ped Mod | Ped Hard | Cyc Mod | Cyc Hard | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline mean±std | 19.3613±0.6280 | 16.3624±0.8570 | 5.4446±0.3652 | 4.3327±0.2982 | 4.4688±0.6479 | 4.1673±0.5800 | 三类原始基线，Car 较稳，Ped/Cyc 波动明显 |
| Full CoP replacement mean±std | 18.9896±0.9216 | 15.9353±0.7208 | 6.2343±0.6636 | 4.8963±0.5455 | 4.0119±0.9861 | 3.6968±0.7612 | Ped 稳定提升，但 Car/Cyc 均值下降，不能作为最终 P2 版本 |
| Full CoP - baseline | -0.3716 | -0.4271 | +0.7897 | +0.5636 | -0.4568 | -0.4705 | 说明直接替换 dim/angle/depth 全属性头风险偏大 |

当前判断：Full CoP replacement 证明链式属性建模对 Pedestrian 有价值，但破坏了 MonoDETR 原有 dim/angle 路径的稳定性，导致 Car 和 Cyclist 均值下降。后续 P2 不再使用 full replacement 作为主方案，改为 Depth-only CoP：保留原始 `inter_references_dim` 和 `angle_embed(hs)`，只用 CoP 链替换 per-query `depth_reg`。

下一轮实验计划：

| 实验名 | 配置 | 状态 | 目标 |
|---|---|---|---|
| CoP-depth-only-run1 | `use_cop=True`, `cop_mode='depth_only'` | 待训练 | 验证是否保留 Ped 增益，同时恢复 Car/Cyc 到 baseline 附近 |
| CoP-depth-only-run2 | 同上，不同随机种子/重复训练 | 待训练 | 判断收益稳定性 |
| CoP-depth-only-run3 | 同上，不同随机种子/重复训练 | 待训练 | 与三次 baseline 做 mean±std 对比 |

## 2026-06-26 阶段性结论记录：Depth-only CoP run1

当前论文主线重新锚定为：面向自动驾驶场景的单目 3D 小目标检测精度改进，不是轻量化。核心问题是小目标像素少、深度线索弱，MonoDETR 的 per-query 深度预测容易受单目深度歧义影响。

当前三个方法层次：

| 模块 | 所属问题 | 改进性质 | 作用位置 | 当前判断 |
|---|---|---|---|---|
| P1 SCA-FPN | 小目标多尺度特征弱 | 特征增强 / neck 改进 | backbone 输出到 transformer 输入前 | 用于提升小目标特征表达，不属于轻量化 |
| Full CoP replacement | 深度与属性预测耦合不足 | 属性预测头重构 | dim/angle/depth 全部替换 | Ped 提升，但 Car/Cyc 均值下降，扰动太大 |
| Depth-only CoP | per-query 深度预测不稳 | 深度头增强 | 只替换 depth_reg，保留原 dim/angle | run1 信号较好，需继续重复验证 |

Depth-only CoP run1 与三次 baseline 均值对比：

| 指标 | baseline mean | Depth-only CoP run1 | 差值 | 初步判断 |
|---|---:|---:|---:|---|
| Car Mod | 19.3613 | 19.4775 | +0.1162 | 基本持平，未明显伤主指标 |
| Car Hard | 16.3624 | 16.1973 | -0.1651 | 小幅下降，仍在 baseline 波动范围内 |
| Ped Mod | 5.4446 | 6.4438 | +0.9992 | 明显提升，是当前最强正信号 |
| Ped Hard | 4.3327 | 5.1207 | +0.7880 | 明显提升，支持小目标叙事 |
| Cyc Mod | 4.4688 | 5.1612 | +0.6924 | 有提升，但需确认稳定性 |
| Cyc Hard | 4.1673 | 4.0974 | -0.0699 | 基本持平 |

当前判断：Depth-only CoP 比 Full CoP replacement 更符合当前论文目标。它没有大幅破坏 Car 主指标，同时明显提升 Pedestrian，Cyclist Moderate 也有正向信号。这个结果说明 P2 的合理定位应是“属性条件引导的 per-query 深度预测增强”，而不是“替换全部属性预测头”。

下一步不应继续盲目加模块。优先完成 Depth-only CoP 的 run2/run3，确认 Pedestrian 的提升是否稳定，同时观察 Car 和 Cyclist 是否保持在 baseline 波动范围内。若三次平均后 Ped Mod/Hard 仍显著高于 baseline，且 Car Mod 不明显下降，则 P2 可以定稿为深度预测头改进。

## 2026-06-29 阶段性结论记录：Depth-only CoP 三次重复实验完成

评估口径保持不变：所有结果均采用 `checkpoint_best.pth`，即按 `Car AP_R40 3D Moderate` 选择 best checkpoint。当前三次 baseline、三次 Full CoP replacement、三次 Depth-only CoP 的统计结论如下。

| 方法 | Car Mod | Car Hard | Ped Mod | Ped Hard | Cyc Mod | Cyc Hard | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline mean±std | 19.3613±0.6280 | 16.3624±0.8570 | 5.4446±0.3652 | 4.3327±0.2982 | 4.4688±0.6479 | 4.1673±0.5800 | 三类原始基线 |
| Full CoP replacement mean±std | 18.9896±0.9216 | 15.9353±0.7208 | 6.2343±0.6636 | 4.8963±0.5455 | 4.0119±0.9861 | 3.6968±0.7612 | Ped 提升，但 Car/Cyc 下降，扰动过大 |
| Depth-only CoP mean±std | 19.8446±0.8441 | 16.3172±0.3251 | 6.6312±0.5641 | 5.0365±0.2055 | 4.6792±0.7122 | 4.1179±0.2878 | 当前 P2 最优方案，可作为主方法 |
| Depth-only CoP - baseline | +0.4833 | -0.0452 | +1.1866 | +0.7038 | +0.2105 | -0.0495 | Car Mod/Ped/Cyc Mod 提升，Hard 基本不伤 |
| Depth-only CoP - Full CoP | +0.8550 | +0.3819 | +0.3969 | +0.1402 | +0.6673 | +0.4210 | 证明 depth-only 比 full replacement 更稳 |

当前判断：Depth-only CoP 可以作为 P2 方向定稿版本。它的改进性质是“深度预测头增强”，不是轻量化。方法动机是：单目 3D 检测中，小目标深度线索弱，原 MonoDETR 的 per-query `depth_embed` 独立预测深度，未充分利用属性条件信息；Depth-only CoP 在保留原始 `dim/angle` 稳定路径的基础上，用 `dim_feat -> angle_feat -> depth_feat` 的链式条件特征增强 per-query `depth_reg`，因此更稳定。

论文表述建议：P2 不写成“完整 CoP 替换属性头”，而写成“属性条件引导的深度预测头增强模块”。Full CoP replacement 可作为消融，说明直接替换全部属性预测头会破坏原模型稳定性；Depth-only CoP 是最终方案。

下一步建议：停止继续修改 P2，优先做 P1 与 P2 的组合实验，即 `SCA-FPN + Depth-only CoP`。如果组合结果稳定，则论文主方法可定为“多尺度特征增强 + 属性条件深度预测增强”的双模块框架。

## 2026-06-29 进展记录：建立 P1+P2 组合分支

当前已从 `P2` 分支新建组合分支：

```text
p1-p2-combined
```

组合方式：只合并训练相关代码，不合并 P1 分支中的 PDF、调研文档等非代码文件。

| 来源分支 | 合并内容 | 作用位置 | 当前状态 |
|---|---|---|---|
| `p1-sca-fpn` | `SCAFPN`、SCA-FPN 配置与 forward 接入逻辑 | `input_proj` 之后，`DepthPredictor` / `Depth-aware Transformer` 之前 | 已接入 |
| `P2` | `CoPDepthHead`、`use_cop`、`cop_mode='depth_only'` | decoder query 输出后的 depth head | 已保留 |

当前组合分支默认配置为：

```yaml
use_sca_fpn: True
sca_fpn_high_guidance: True
use_dcn_lateral: True
use_cop: True
cop_mode: 'depth_only'
```

静态检查结果：

- `monodetr.py` 与 `sca_fpn.py` 通过 `python -m py_compile`。
- `configs/monodetr.yaml` 通过 yaml 解析。
- `git diff --check` 未发现 whitespace error。

下一步实验建议：在服务器上直接使用 `p1-p2-combined` 分支的 `configs/monodetr.yaml` 跑第一次 `P1 + P2` 组合训练。建议至少重复三次。注意：此处原计划仍沿用 `Car AP_R40 3D Moderate` 选 best，但 2026-06-29 后续记录已将标准统一改为三类 Moderate R40 3D 均值，后续实验以新标准为准。


| 这组数据，作为和MonoDETR论文中的Val AP数据进行对比，证明本地环境、数据集划分、训练配置和 MonoDETR 官方结果基本对齐 |                                                                                                                                |                                     |      |         |          |         |   |
|:------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------:|:----:|:-------:|:--------:|:-------:|---|
|                                                      实验名称                                                      |                                                            配置说明                                                            |                 类别                | 指标 |   Easy  | Moderate |   hard  |   |
|                                                      baseline                                                      |                                                  按照car moderate选best epoch                                                  |     Car AP_R40@0.70, 0.70, 0.70:    |  3d  | 28.0994 |  19.9693 | 17.3196 |   |
|                                                                                                                    |          日志路径：/desay120T/ct/dev/uid01955/MonoDETR-P2-useK/jieguo/monodetr_basline_run2/train.log.20260624_194135          | Pedestrian AP_R40@0.50, 0.50, 0.50: |  3d  |  7.1964 |  5.2696  |  4.2265 |   |
|                                                                                                                    |                                                                                                                                |   Cyclist AP_R40@0.50, 0.50, 0.50:  |  3d  |  7.6784 |  3.8903  |  3.5413 |   |
|                                                                                                                    |                                                                                                                                |                                     |      |         |          |         |   |
|     这组数据，用的是和上面这个同一个训练结果，但是改成按三类均值选择best_epch，后续所有的改进对比都对比这组数据    |                                                                                                                                |                                     |      |         |          |         |   |
|                                                      实验名称                                                      |                                                            配置说明                                                            |                 类别                | 指标 |   Easy  | Moderate |   hard  |   |
|                                                      baseline                                                      |                                                     按照3类均值选best epoch                                                    |     Car AP_R40@0.70, 0.70, 0.70:    |  3d  | 27.9215 |  19.7731 | 16.5456 |   |
|                                                                                                                    |          日志路径：/desay120T/ct/dev/uid01955/MonoDETR-P2-useK/jieguo/monodetr_basline_run2/train.log.20260624_194135          | Pedestrian AP_R40@0.50, 0.50, 0.50: |  3d  |  7.8701 |  5.7994  |  4.5808 |   |
|                                                                                                                    |                                                                                                                                |   Cyclist AP_R40@0.50, 0.50, 0.50:  |  3d  |  8.3014 |  4.3058  |  3.8296 |   |
|                                                                                                                    |                                                                                                                                |                                     |      |         |          |         |   |
|                                             P2-CoP-depth_only——改进数据                                            |                                                                                                                                |                                     |      |         |          |         |   |
|                                                      实验名称                                                      |                                                            配置说明                                                            |                 类别                | 指标 |   Easy  | Moderate |   hard  |   |
| P2-CoP-depth_only                                                                                                  |                                                     按照3类均值选best epoch                                                    |     Car AP_R40@0.70, 0.70, 0.70:    |  3d  | 26.5177 |  18.9833 | 15.9105 |   |
|                                                                                                                    |       日志路径：/desay120T/ct/dev/uid01955/MonoDETR-P2-useK/jieguo/monodetr_CoP_depth_only_run1/train.log.20260625_195352      | Pedestrian AP_R40@0.50, 0.50, 0.50: |  3d  | 10.6904 |  8.0000  |  6.6025 |   |
|                                                                                                                    |                                                              run1                                                              |   Cyclist AP_R40@0.50, 0.50, 0.50:  |  3d  | 10.0810 |  5.1866  |  4.5886 |   |
|                                                                                                                    |                                                                                                                                |                                     |      |         |          |         |   |
|                                                      实验名称                                                      |                                                            配置说明                                                            |                 类别                | 指标 |   Easy  | Moderate |   hard  |   |
| P2-CoP-depth_only                                                                                                  |                                                     按照3类均值选best epoch                                                    |     Car AP_R40@0.70, 0.70, 0.70:    |  3d  | 27.6673 |  20.4837 | 16.4533 |   |
|                                                                                                                    |       日志路径：/desay120T/ct/dev/uid01955/MonoDETR-P2-useK/jieguo/monodetr_CoP_depth_only_run2/train.log.20260626_170953      | Pedestrian AP_R40@0.50, 0.50, 0.50: |  3d  |  8.8936 |  6.7671  |  5.0126 |   |
|                                                                                                                    |                                                              run2                                                              |   Cyclist AP_R40@0.50, 0.50, 0.50:  |  3d  |  9.8842 |  5.3119  |  5.0960 |   |
|                                                                                                                    |                                                                                                                                |                                     |      |         |          |         |   |
|                                                      实验名称                                                      |                                                            配置说明                                                            |                 类别                | 指标 |   Easy  | Moderate |   hard  |   |
| P2-CoP-depth_only                                                                                                  |                                                     按照3类均值选best epoch                                                    |     Car AP_R40@0.70, 0.70, 0.70:    |  3d  | 26.3028 |  19.0884 | 15.9506 |   |
|                                                                                                                    |       日志路径：/desay120T/ct/dev/uid01955/MonoDETR-P2-useK/jieguo/monodetr_CoP_depth_only_run3/train.log.20260627_111011      | Pedestrian AP_R40@0.50, 0.50, 0.50: |  3d  |  8.9820 |  7.1128  |  5.2569 |   |
|                                                                                                                    |                                                              run3                                                              |   Cyclist AP_R40@0.50, 0.50, 0.50:  |  3d  |  9.3989 |  4.6388  |  3.8912 |   |
|                                                                                                                    |                                                                                                                                |                                     |      |         |          |         |   |
|                                   P1-SCA-FPN-Highguidance=True-DCN=True——改进数据                                  |                                                                                                                                |                                     |      |         |          |         |   |
|                                                      实验名称                                                      |                                                            配置说明                                                            |                 类别                | 指标 |   Easy  | Moderate |   hard  |   |
| P1-SCA-FPN-Highguidance=True-DCN=True                                                                              |                                                     按照3类均值选best epoch                                                    |     Car AP_R40@0.70, 0.70, 0.70:    |  3d  | 25.8875 |  18.9672 | 15.8594 |   |
|                                                                                                                    | 日志路径：/desay120T/ct/dev/uid01955/MonoDETR-main/jiegou/P1-SCA-FPN-Highguidance=True-DCN=True-Run2/train.log.20260528_095341 | Pedestrian AP_R40@0.50, 0.50, 0.50: |  3d  |  8.9627 |  7.0836  |  5.3242 |   |
|                                                                                                                    |                                                  run2，注：第一次没训练完断了                                                  |   Cyclist AP_R40@0.50, 0.50, 0.50:  |  3d  | 10.0186 |  5.5273  |  4.3425 |   |
|                                                                                                                    |                                                                                                                                |                                     |      |         |          |         |   |
|                                                      实验名称                                                      |                                                            配置说明                                                            |                 类别                | 指标 |   Easy  | Moderate |   hard  |   |
| P1-SCA-FPN-Highguidance=True-DCN=True                                                                              |                                                     按照3类均值选best epoch                                                    |     Car AP_R40@0.70, 0.70, 0.70:    |  3d  | 27.1302 |  19.3293 | 16.1102 |   |
|                                                                                                                    | 日志路径：/desay120T/ct/dev/uid01955/MonoDETR-main/jiegou/P1-SCA-FPN-Highguidance=True-DCN=True-Run3/train.log.20260603_102115 | Pedestrian AP_R40@0.50, 0.50, 0.50: |  3d  | 10.9379 |  8.2033  |  6.6457 |   |
|                                                                                                                    |                                                              run3                                                              |   Cyclist AP_R40@0.50, 0.50, 0.50:  |  3d  | 12.4880 |  6.5201  |  5.8428 |   |
|                                                                                                                    |                                                                                                                                |                                     |      |         |          |         |   |
|                                                      实验名称                                                      |                                                            配置说明                                                            |                 类别                | 指标 |   Easy  | Moderate |   hard  |   |
| P1-SCA-FPN-Highguidance=True-DCN=True                                                                              |                                                     按照3类均值选best epoch                                                    |     Car AP_R40@0.70, 0.70, 0.70:    |  3d  | 27.3224 |  19.4730 | 16.1925 |   |
|                                                                                                                    | 日志路径：/desay120T/ct/dev/uid01955/MonoDETR-main/jiegou/P1-SCA-FPN-Highguidance=True-DCN=True-Run4/train.log.20260604_104817 | Pedestrian AP_R40@0.50, 0.50, 0.50: |  3d  | 10.2740 |  7.5307  |  6.0640 |   |
|                                                                                                                    |                                                              run4                                                              |   Cyclist AP_R40@0.50, 0.50, 0.50:  |  3d  |  8.8672 |  4.7641  |  4.0494 |   |

## 2026-06-29 进展记录：修改 best checkpoint 选择标准

根据新增实验记录，后续所有训练与对比统一使用三类 Moderate R40 3D 均值选择 `checkpoint_best.pth`：

```text
best_selection_score =
(Car_3d_moderate_R40 + Pedestrian_3d_moderate_R40 + Cyclist_3d_moderate_R40) / 3
```

已完成代码修改：

| 文件 | 修改内容 |
|---|---|
| `lib/datasets/kitti/kitti_dataset.py` | `eval()` 不再返回单独 `Car_3d_moderate_R40`，改为返回三类 `3d_moderate_R40` 均值 |
| `lib/helpers/trainer_helper.py` | 日志文案从 `Best Result` 改为 `Best Selection Result`，避免误解 |
| `configs/monodetr.yaml` | 注释说明 `writelist` 必须包含 `Car/Pedestrian/Cyclist`，因为 best 选择依赖三类均值 |
| `P1_P2_combination_summary.md` / `P2_depth_only_cop_summary.md` | 同步更新后续实验口径 |

注意：早期实验表中“按 Car Moderate 选 best”的结果仍保留为历史记录；从本条记录之后，论文主实验和后续改进实验都应以三类均值选 best 的结果为准。
