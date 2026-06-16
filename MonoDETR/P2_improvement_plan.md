# P2 改进方向完整方案

## 一、方向一 P1 SCA-FPN 回顾与合理性分析

### 1.1 P1 做了什么

P1（Spatial-Channel Attention FPN + DCN lateral）在 MonoDETR 的 `input_proj` 阶段原地改写多尺度特征 `srcs`，核心操作：

1. 通道注意力 (CA)：CBAM 风格，每层独立 AvgPool+MaxPool → 共享 MLP (reduction=16)
2. 局部空间注意力 (SA)：channel-wise avg+max → 7×7 Conv
3. 高层引导：最高层 (1/64) 的 SA 上采样到低层，与低层 SA 取均值——"高语义指导低层"
4. DCN lateral：1×1 Conv+GN 升级为 3×3 DeformConv+GN，offset 预测卷积 zero-init
5. 残差连接：`outs[i] = lateral(gated) + srcs[i]`

### 1.2 P1 的合理性分析——强

P1 打的短板是真实存在的。MonoDETR 的 `input_proj` 对 4 级特征 `{1/8, 1/16, 1/32, 1/64}` 只做 1×1 卷积通道对齐，没有任何跨尺度语义交互。同时 `depth_predictor` 内部的多尺度融合是平凡求和 `(src_8 + src_16 + src_32) / 3`。对于论文关心的远距离小目标（Pedestrian Hard, Cyclist Hard），在 1/32 尺度上已被压到不足 2 像素——浅层有几何信息但缺语义，深层有语义但缺分辨率。

P1 的改进思路在方法论上是健康的：

第一，高层引导设计（只在最高层算 SA，然后上采样指导低层）比 BiFPN/PaFPN 的对称双向融合更适合 3D 检测。原因：3D 检测的目标响应本身就稀疏（全图最多几十个物体），不像语义分割需要密集的逐像素预测。对称融合会把高层稀疏响应平均掉，而单向的"老师指导学生"模式保留了高层的强判别力。

第二，DCN lateral 的零初始化设计很聪明——offset 预测 conv 权重初始化为 0，iter 0 退化为标准 3×3 卷积，对训练统计量零扰动，然后在训练中逐步学习贴合目标真实形状的采样点。这对于小/遮挡/截断目标尤其有意义——标准 3×3 卷积的固定方形感受野可能覆盖大量背景，而 DeformConv 可以"贴着物体的轮廓"采样。

第三，残差连接 `outs[i] = lateral(gated) + srcs[i]` 确保即使注意力权重还没学好，也不会倒退到比 baseline 更差——这是与 P2 几何融合（首版无门→检测器崩溃到 Car Mod 10.9）形成鲜明对比的正确工程实践。

### 1.3 P1 的潜在局限

从架构角度看，P1 有两个可以讨论的点：

第一，通道注意力的 reduction=16 是否过度压缩了信息——对于 256 维的 hidden_dim，压缩到 16 维意味着只保留了 6.25% 的信息维度。这在 2D 检测中效果很好（CBAM 原论文就是 reduction=16），但 3D 检测的深度信息编码更加密集（256 维里同时编码了空间位置、语义、深度等异构信息），16 维的 bottleneck 可能损失对深度敏感的特征。不过这不是致命问题——如果 P1 确实涨点了，说明信息损失在可接受范围内。

第二，高层引导只用了单层（1/64）。FPN 的经典发现是"每一层都可以从相邻层获益"——仅用最高层指导所有低层，中间的 1/32 层的独特信息（刚好对应中等距离目标的最优尺度）没有向上传递。但这是设计选择而非缺陷——MonoDETR 的深度预测器本身就融合了 1/8、1/16、1/32 三层，1/64 在 baseline 里并没有被深度预测器使用，P1 把 1/64 作为"老师"层是一个合理的选择。

### 1.4 P1 在论文中的定位

P1 作为第一个改进方向，其定位是"特征增强"——在进入深度预测和 Transformer 之前，让特征本身更丰富。这在论文逻辑上是合理的：先改进特征质量（P1），再改进深度预测质量（P2），最后改进解码器的查询交互质量（P3），按数据流方向逐步优化。

---

## 二、P2（原 DBDU）的经验教训

### 2.1 原 P2 做了什么

原 P2 在 p2-dynamic-bins 分支上尝试了两个机制：

**机制 1：Dynamic Bins（动态深度桶）**
- 想法：将全图共享的固定 LID 80 桶改为每图自适应的深度分桶
- 第一版（自由 softmax）：桶塌缩，80 桶中约 60 个挤死在 56-60m
- 第二版（tanh 有界）：塌缩修好，但 tanh 饱和使 per-image 退化回全局静态桶
- 结果：三列 AP 全落在 baseline 噪声带，零增益

**机制 2：几何深度 σ 加权融合**
- 想法：将 `(reg + geo + map) / 3` 朴素平均改为按精度逆方差加权
- 首版（无安全门）：检测器崩溃，Car Mod 10.9
- 第二版（零初始化 sigmoid 门）：门焊死，等于又跑一遍 baseline
- 第三版（clamp-linear 门 + loss-σ 解耦）：安全但 AP 仍在 baseline 噪声带内

### 2.2 核心教训

两个机制失败的共同根因已经坐实：**它们都只能间接影响最终 3D 框深度，而最终深度由 per-query `depth_embed` 头主导。**

具体来说：
- Dynamic Bins 改的 bin 中心只经 `weighted_depth → 深度位置编码 + 前景深度图 aux loss` 间接影响 decoder，decoder 的 per-query depth_embed 头完全不受 bins 约束
- 几何 σ 加权融合只是重新分配三个已有深度源的权重，没有任何新的表达力——如果三个源本身在这个场景下都不准，怎么加权都不会突然变准

这个教训对 P2 的新方向有直接的指导意义：**必须上直接作用于 per-query `depth_embed` 头的强杠杆**，不能再用"调整已有源"的间接思路。

---

## 三、P2 新方向：RARE-style K-假设深度预测

### 3.1 改进思路

MonoDETR 的 `depth_embed` 头目前对每个 object query 只输出一个深度假设：`[depth_reg, log_variance]`（2 维）。这个单点估计在面对单目 3D 检测的核心难题——2D 像素到 3D 深度的多义性——时极为脆弱。同一个 2D 框可以对应无数个不同深度的 3D 物体，而只有一个深度预测意味着模型被迫在训练中学习"平均"的深度映射，这对小目标（像素少→视觉歧义大）尤其不利。

RARE（CVPR 2026）的核心洞察是：与其逼模型输出一个"最好的"深度，不如让模型输出 K 个"可能的"深度假设，然后通过学习排名（ranking）来选择最优假设。这一思路直接命中单目 3D 检测的根本难题。

### 3.2 来源论文

RARE: Learn to RAnk and REtrieve for Monocular 3D Object Detection
- 会议：CVPR 2026 (Highlight)
- 作者：Hyeonjeong Park, Peixi Xiong 等
- 代码：https://github.com/HyeonjeongPark37/RARE
- KITTI 结果：Car AP3D Mod. 19.57（+4.5% over MonoDGP），Cyclist AP3D Mod. 5.96（+39%）

### 3.3 实现原理

将 depth_embed head 从输出 1 个假设改为输出 K 个假设：

```
原来：MLP(256 → 256 → 2)   → [mean, log_var]  (1个深度假设)
改为：MLP(256 → 256 → K×2)  → [mean_1, var_1, ..., mean_K, var_K]  (K个深度假设)
```

每个假设的回归深度与 `d_geo`、`d_map` 独立做三源融合：`d_fused_k = (d_reg_k + d_geo + d_map) / 3`。`d_geo` 和 `d_map` 对所有 K 个假设是相同值，仅贡献一个共享偏置，假设间差异完全由回归头产生（`d_fused_i - d_fused_j = (d_reg_i - d_reg_j) / 3`），多样性不受影响。这样既保留了 MonoDETR 的核心深度融合机制，又引入了多假设覆盖深度歧义的能力。

推理时从 K 个假设中选择与最终 3D 框最匹配的那个（通过置信度排序或直接选择置信度最高的假设）。

关键配套设计：

**多样性正则化（防止 K 个假设坍缩到同一个值）**
- 对 K 个 mean 计算 pairwise L2 距离，激励它们覆盖不同的深度区间
- 或者用 winner-take-all 训练：每个 GT depth 只匹配 K 个假设中最近的那个，只对该假设回传梯度

**置信度排序损失（Learn to Rank）**
- 将分类头的置信度改为不仅预测"是否为目标"，还要预测"哪个深度假设更可靠"
- 用 pairwise ranking loss：对于同一 query 的 K 个假设，鼓励置信度排序与深度误差排序一致
- 具体：如果假设 k1 的深度 error < 假设 k2 的深度 error，则置信度 score_k1 应该 > score_k2

**推理时检索（Learn to Retrieve）**
- 对每个 query，选择置信度最高的假设对应的深度
- 或者对所有 K 个假设的 3D 框做 NMS（如果 K 较小）

### 3.4 具体改动清单

（列出具体代码文件、函数、行数，以及要改动的内容）

**改动 1：`monodetr.py` — `depth_embed` 头**

`__init__` 中：
```python
# [P2] K-hypothesis depth prediction
self.K = 4  # 假设数量
self.depth_embed = nn.ModuleList([
    MLP(hidden_dim, hidden_dim, self.K * 2, 2)
    for _ in range(num_pred)
])
```
- 原 `MLP(256, 256, 2, 2)` 改为 `MLP(256, 256, K*2, 2)`
- 输出从 `[B, NQ, 2]` 变为 `[B, NQ, K*2]`，reshape 为 `[B, NQ, K, 2]`

**改动 2：`monodetr.py` — `forward` 中的 `depth_ave` 计算**

每个假设独立做三源深度融合，保持 MonoDETR 的核心机制不变。`d_geo` 和 `d_map` 是确定性值（每个 query 只有一个），但它们对所有 K 个假设只贡献同一个共享偏置，不会破坏假设间相对差异：

```
d_fused_k = (d_reg_k + d_geo + d_map) / 3
```

K 个假设之间的差异完全由回归头产生：`d_fused_i - d_fused_j = (d_reg_i - d_reg_j) / 3`，多样性保留。因此不需要牺牲深度融合。

```python
# [P2-KHYP] K-hypothesis with three-source fusion preserved
depth_reg = self.depth_embed[lvl](hs[lvl])  # [B, NQ, K*2]
depth_reg = depth_reg.view(B, NQ, self.K, 2)
# 每个 hypothesis 的回归 depth: 1/sigmoid(d[:,:,:,0]) - 1 => [B, NQ, K]
depth_hyps = 1.0 / (depth_reg[..., 0].sigmoid() + 1e-6) - 1.0
log_vars  = depth_reg[..., 1]               # [B, NQ, K]
# 三源深度融合：每个假设独立与 geo/map 融合
depth_fused = (depth_hyps + depth_geo.unsqueeze(-1) + depth_map.unsqueeze(-1)) / 3  # [B, NQ, K]

# depth_ave 构造：不能简单取第一个假设（WTA 选中的可能不是它）
# 用假设置信度加权平均，训练初期（conf 未训练）退化为均匀平均≈baseline /3
hypo_conf = F.softmax(self.hypo_conf_embed[lvl](hs[lvl]), dim=-1)  # [B, NQ, K]
depth_ave_val = (hypo_conf * depth_fused).sum(dim=-1)               # [B, NQ]
log_var_ave   = (hypo_conf * log_vars).sum(dim=-1)                  # [B, NQ]
depth_ave = torch.stack([depth_ave_val, log_var_ave], dim=-1)       # [B, NQ, 2]
```

**改动 3：`monodetr.py` — 置信度头**

需要新增一个 "假设选择" 子头：
```python
# [P2] hypothesis confidence scorer
self.hypo_conf_embed = nn.ModuleList([
    MLP(hidden_dim, hidden_dim, self.K, 2)
    for _ in range(num_pred)
])
```
输出 `[B, NQ, K]`，经 softmax 后表示每个假设被选中的可靠性。

**改动 4：`SetCriterion` — 深度损失**

原 `loss_depths` 使用拉普拉斯不确定性损失：
```python
loss = 1.4142 * exp(-log_var) * |pred - gt| + log_var
```

改为 winner-take-all（基于融合后的深度，保持与三源融合一致）：

```python
# [P2-KHYP] Winner-Take-All loss
errors = abs(depth_fused - depth_gt.unsqueeze(-1))  # [B, NQ, K]

# ε-greedy 防饿死：10% 概率随机选假设，确保 K 个都有机会被训练
if self.training and random.random() < 0.1:
    best_k = torch.randint(0, self.K, (B, NQ), device=errors.device)
else:
    best_k = errors.argmin(dim=-1)

# 只对 best_k 对应的假设算 loss
gather_idx = best_k.unsqueeze(-1)
depth_pred_best = depth_fused.gather(-1, gather_idx).squeeze(-1)  # [B, NQ]
log_var_best   = log_vars.gather(-1, gather_idx).squeeze(-1)      # [B, NQ]
loss_depth = 1.4142 * exp(-log_var_best) * abs(depth_pred_best - depth_gt) + log_var_best
```

可选升级——Soft WTA（更平滑，用温度系数控制选择硬度）：
```python
# 误差越小权重越高，temperature=0.1 → 近似硬选择，temperature=1.0 → 软平均
weights = F.softmax(-errors / temperature, dim=-1)  # [B, NQ, K]
loss_depth = (weights * laplacian_nll(depth_fused, log_vars, depth_gt.unsqueeze(-1))).sum(dim=-1)
```
```python
# 鼓励不同假设之间的融合深度值分散
depth_fused_pairwise_diff = |depth_fused[:,:,:,None] - depth_fused[:,:,None,:]|  # [B, NQ, K, K]
loss_diversity = -mean(depth_fused_pairwise_diff) / (K * (K-1))
```
或者用更稳定的方法——鼓励各假设的 mean 在合理范围内均匀分布（如 bins=K 等间距的 L1 对齐 loss）。

**改动 5：`decode_helper.py` — 推理时假设选择**

```python
# 推理时选择置信度最高的假设（使用融合深度）
hypo_conf = softmax(hypo_conf_logits, dim=-1)  # [B, NQ, K]
best_k = argmax(hypo_conf, dim=-1)
depth_selected = depth_fused[..., best_k]  # 融合深度，非裸回归深度
```

### 3.5 预期效果

RARE 在 KITTI 上的核心增益来自两方面：(1) 多假设覆盖了深度歧义的不同解，`winner-take-all` 让每个假设专精一个深度区间；(2) Cyclist 类提升最大（+39%），因为骑行者形状细长、像素极少，深度歧义最严重——恰好命中论文"小目标"叙事。

预期 P2 单此一项能在 MonoDETR baseline 上获得 1-3 点 Car Mod 提升，Ped/Cyc Hard 提升可能更显著（因为这些类别的深度歧义更大）。

### 3.6 风险与缓解

- K 个假设可能坍缩到同一点：diversity loss 是必须的，且需要合理的权重（建议从 1.0 开始调）
- 训练初期所有假设都不准，winner-take-all 会导致某些假设"饿死"：可以用 ε-greedy 策略（10% 概率随机选假设训练）或 soft winner-take-all（对前 M 个最近假设分配递减权重）
- 增加的计算量：K×2 维输出替代 2 维，增加量极小（每个假设只增加 2 个标量输出）

---

## 四、P2 实验路线图

### 4.1 第一轮：RARE K-假设深度预测（核心创新，需代码实现）

1. 在 baseline 代码基础上实现 K-假设 depth_embed，保持三源深度融合
2. 先做 r1（winner-take-all loss），观察 K 个假设是否有效分化
3. 如果假设坍缩，加 diversity loss
4. 如果分化良好，加 Learn to Rank
5. 每次修改后跑完整 195 epoch

### 4.2 消融实验设计

| 行 | 配置 | 目的 |
|---|---|---|
| r0 | baseline | 原 MonoDETR（depth_embed 输出 2 维） |
| r0.5 | K=1，但网络结构改为 K×2 输出 | 排除"网络变宽"因素，验证是 K 假设起作用非参数量增加 |
| r1 | K=4, WTA (ε-greedy), 无 diversity | 多假设 + 只训练最近假设 |
| r2 | r1 + diversity loss | + 多样性正则化，防坍缩 |
| r2.5 | K=2 和 K=6 对比 | 验证最优 K 值 |
| r3 | r2 + ranking loss (Learn to Rank) | + 置信度排序学习 |
| r4 | r3 + Soft WTA (temperature=0.1) | 从硬 WTA 升级为软 WTA |
| r5 | r4 最终版 | 完整 RARE-style |

### 4.3 可视化验证（必须输出）

**假设分化曲线**：横轴训练 epoch，纵轴 4 个假设的 batch 平均预测深度。预期：4 条线从重合逐渐分离，各自覆盖不同的深度区间。这直接证明 K 假设在"分工"而非坍缩——是说服审稿人的核心证据。

**假设-目标匹配热图**：横轴目标真实深度（0-80m 分 8 bin），纵轴 4 个假设，颜色为该假设被 WTA 选中的频率。预期：假设 1 高频命中近处目标（0-20m），假设 4 高频命中远处目标（>40m）。说明假设确实按深度区间分工。

### 4.4 第二轮：如果 RARE 方向正反馈

1. 将 RARE 的假设置信度与原有的 class 置信度联合使用
2. 在 NMS/后处理中利用多假设信息（如：对于高质量检测，使用假设中的 consensus depth；对于低质量检测，使用置信度加权平均）

### 4.5 如果 RARE 方向没有正反馈（需要备选方案）

备选方案 A：MonoCoP（CVPR 2026）链式预测。将 depth_embed head 从独立 MLP 改为以朝向特征为条件：
```python
# 链式预测: dim_feat → angle_feat → depth_feat
dim_feat   = MLP(hs)(hs)          # 尺寸特征
angle_feat = MLP(dim_feat)(dim_feat)  # 朝向特征（以尺寸为条件）
depth_feat = MLP(cat(angle_feat, hs))(cat(angle_feat, hs))  # 深度特征（以朝向为条件）
depth_out  = Linear(depth_feat)   # [mean, log_var]
```
改动量小（仅修改 depth_embed 内部结构），与 DINO baseline 兼容。

备选方案 B：Salience DETR（CVPR 2024）尺度无关显著性。在 decoder 的 self-attention 或分类头中加入尺度无关的显著性监督，确保小目标不被 50 个 query 中淹没。主要改动在 `SetCriterion.loss_labels` 中引入尺度相关的 focal weight。

---

## 五、论文叙事框架

### P2 作为论文第三章（或论文核心创新点）

标题建议：《基于多假设排序的单目 3D 检测深度优化方法》

章节结构：
1. 引言：单目 3D 检测的深度歧义问题——同一 2D 框对应无数个深度可能，单点估计在遮挡/远距离/小目标场景下不可靠
2. 方法：
   - 2.1 基线回顾：MonoDETR 的 depth_embed 头与深度融合机制
   - 2.2 K 假设生成：depth_embed 头扩展为 K 个独立深度预测
   - 2.4 学习排序：通过假设置信度预测和 ranking loss 选择最优深度
   - 2.5 多样性正则化：防止假设坍缩
3. 实验：消融实验（K 的选择、diversity loss 权重、winner-take-all vs soft selection）+ KITTI 主结果
4. 分析：为什么多假设在 Cyclist/Pedestrian 上提升最大（像素少→歧义大）→ 支撑"小目标"论文叙事

### 论文方法差异化叙述（必须明确）

论文中需要清晰区分本方法与其他深度不确定性方法。核心对比：

MonoDETR 原生的 Laplacian NLL 通过 log-variance 输出"我不确定"的信号，本质上是被动承认不确定性，仍是从单一预测出发的"判断题"——给出一个深度 + 一个不确定度，但无法覆盖深度空间的多个峰。

本文的 K-假设机制将范式从"判断题"升级为"选择题"——模型主动生成 K 个候选深度覆盖歧义空间，通过学习排序选择最优解。这本质上是将单峰拉普拉斯分布升级为 K-混合分布，每个分量专精一个深度区间。从信息论角度，单峰分布在多峰深度歧义场景下必然丢失信息，K-混合分布能将歧义保留到最终决策阶段。K 个假设不同于简单的网络变宽（可以通过 r0.5 消融实验排除参数量增加的干扰），也不同于 ensemble（共享特征 backbone、一个 head 输出 K 个假设 vs K 个独立模型）。

---

## 六、关键实现注意事项

基于 P2 第一次尝试的教训，以下原则必须在实现中遵守：

1. **config 开关控制**：所有新功能用 `use_k_hypothesis`、`num_hypotheses`、`use_hypo_rank_loss`、`hypo_diversity_weight` 等 config 字段控制，默认 False 时精确回到 baseline。
2. **cfg.get() 读取**：所有新 config 字段用 `cfg.get('key', default)` 读取，保证旧 yaml 可加载。
3. **注释 tag**：所有新代码用 `# [P2-KHYP]` 标注。
4. **评测纪律**：采用论文标准做法，以 `checkpoint_best`（按 Car Mod AP3D R40 选帧）的结果作为最终汇报数据。内部消融判断时注意：Ped/Cyc 在 checkpoint_best 帧的数值受选帧噪声影响（旧 P2 实验中曾出现 Ped Hard 单点 +1.6 但实为噪声），建议结合多 seed 重复实验确认增益的稳定性和一致性，确保论文数据的可信度。

---

## 七、改动文件清单

| 文件 | 改动 | tag |
|---|---|---|
| `configs/monodetr.yaml` | 新增 `use_k_hypothesis` / `num_hypotheses` / `use_hypo_rank_loss` / `hypo_diversity_weight` | `# [P2-KHYP]` |
| `lib/models/monodetr/monodetr.py` — `MonoDETR.__init__` | `depth_embed` 改为 `MLP(256, 256, K*2, 2)`；新增 `hypo_conf_embed` | `# [P2-KHYP]` |
| `lib/models/monodetr/monodetr.py` — `MonoDETR.forward` | depth_ave 计算改为 K 假设 + reshape | `# [P2-KHYP]` |
| `lib/models/monodetr/monodetr.py` — `SetCriterion.loss_depths` | winner-take-all + diversity loss | `# [P2-KHYP]` |
| `lib/models/monodetr/monodetr.py` — `SetCriterion.forward` | 新增假设损失 weight | `# [P2-KHYP]` |
| `lib/helpers/decode_helper.py` | 推理时 K 假设选择逻辑 | `# [P2-KHYP]` |
