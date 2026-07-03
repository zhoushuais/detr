# P2-DBDU 方向二设计文档（第一增量：Dynamic Bins）

> 本文档对应论文方向二 **DBDU = Dynamic Bins + Depth Uncertainty + ADPM 残差精修** 的**第一个增量**。
> 第一增量**只做 Dynamic Bins 核心**，其余子部件（ADPM 残差头 / 几何深度融合 / log-z 压缩 / L_ud 不确定性损失）全部留到后续增量。
> 分支：`p2-dynamic-bins`（从 `v0-baseline` 派生）。注释 tag：`# [P2-DBDU]`。

## 1. 打的短板

MonoDETR 的前景深度分支用**固定 LID 80 桶**离散化深度，桶边界在 [depth_predictor.py](lib/models/monodetr/depth_predictor/depth_predictor.py) 构造时算死、`requires_grad=False`、**所有图像共享同一套桶**：

```python
# depth_predictor.py:21-25（baseline）
bin_size  = 2 * (depth_max - depth_min) / (depth_num_bins * (1 + depth_num_bins))
bin_value = (bin_indice + 0.5).pow(2) * bin_size / 2 - bin_size / 8 + depth_min
self.depth_bin_values = nn.Parameter(bin_value, requires_grad=False)   # 固定、全图共享
```

LID 桶虽然让远处桶变宽（缓解远距离误差），但**对所有场景一刀切**：一张近景拥挤图和一张空旷远景图用的是同一套桶边界。深度分布逐图不同，固定桶无法自适应。

## 2. 核心做法：AdaBins 风格每图自适应桶

把"固定桶中心"换成"每图预测的桶中心"，**分类头输出维度不变（仍 `num_bins+1`）**，只换桶中心的数值来源。

**BinPredictor（新增轻量头，加在 `DepthPredictor`）**：

1. 取 `depth_head` 输出的特征 `src (B, C=256, H, W)`，`AdaptiveAvgPool2d(1)` → `(B, 256)`。
2. `MLP(256 → 256 → num_bins)` → 每图 `num_bins` 个 raw width。
3. `softmax(width, dim=-1)` → 归一化（每图桶宽之和=1）→ × `(depth_max − depth_min)` → 每桶实际宽度。
4. `cumsum` → 桶边界 `edges (B, num_bins+1)`，从 `depth_min` 到 `depth_max`；桶中心 = 相邻边界中点 `centers (B, num_bins)`，末尾补 `depth_max` 凑齐 `num_bins+1` 以对齐分类头通道。

**为什么不需要单调约束损失**：`softmax` 保证桶宽恒正，`cumsum` 保证边界严格单调递增且覆盖满量程 `[depth_min, depth_max]`——单调性由构造保证，计划里的"Bins 顺序约束损失"在本增量直接省掉（YAGNI）。

**iter 0 平滑起步（zero-init，借 P1 的 DCN 思路）**：`BinPredictor` 末层 linear 用 **zero 权重 + LID 形状的 bias**（`bias = log(normalized LID widths)`，LID 桶宽 ∝ `(i+1)`）。于是 `iter 0` 时 `softmax(bias)` = 归一化 LID 桶宽、与输入无关，**每张图都精确复现固定 LID 分桶**；随训练权重离开 0，桶才逐步变成每图自适应。对 baseline 统计量零扰动起步，不会一上来就把深度分桶打乱。

## 3. 关键集成点：桶被编码在**两个**地方

深度桶不止用在前向，GT 深度做深度图损失时也要按桶离散化。两处都得切到每图桶，否则前向与监督目标错位：

| 位置 | 文件 | baseline 行为 | 动态桶改动 |
|---|---|---|---|
| **前向** weighted_depth | [depth_predictor.py:77](lib/models/monodetr/depth_predictor/depth_predictor.py#L77) | `Σ(probs × depth_bin_values)` | 换成每图 `centers` |
| **损失** GT→桶索引 | [ddn_loss.py:66-102](lib/models/monodetr/depth_predictor/ddn_loss/ddn_loss.py#L66) `bin_depths()` | LID **闭式反函数**解析求索引 | 动态桶无闭式 → `torch.bucketize` 按每图 `edges` 离散化 |

**bucketize 是本增量的核心工作量**：原 `bin_depths()` 用 LID 解析公式 `indices = -0.5 + 0.5*sqrt(1 + 8*(d-dmin)/bin_size)` 直接算桶号；动态桶边界是任意的、没有闭式，必须逐 batch 元素用该图的 `edges` 对 GT 深度做 `torch.bucketize`/`searchsorted`，得到目标桶号。

## 4. 改动清单（4 个文件）

- **[depth_predictor.py](lib/models/monodetr/depth_predictor/depth_predictor.py)**
  - `__init__`：读 `use_dynamic_bins` / `num_dynamic_bins`；动态时构造 `BinPredictor`（MLP 头）；固定桶 `depth_bin_values` 保留作 fallback。
  - `forward`：动态时算每图 `centers/edges`，`weighted_depth` 用 `centers`；返回值**多带 `bin_edges`** 供损失用（`use_dynamic_bins=False` 时返回 `None`）。
- **[ddn_loss.py](lib/models/monodetr/depth_predictor/ddn_loss/ddn_loss.py)**
  - `forward` / `bin_depths` 增加 `bin_edges=None` 入参；`None` → 走原 LID 闭式（逐位不变）；非 `None` → 逐图 `bucketize`。
- **[monodetr.py](lib/models/monodetr/monodetr.py)**
  - `forward`：接收 depth_predictor 多返回的 `bin_edges`，存进 `out['pred_depth_bins']`。
  - `SetCriterion.loss_depth_map`：读 `outputs.get('pred_depth_bins')` 传给 `self.ddn_loss`。
- **[configs/monodetr.yaml](configs/monodetr.yaml)**：新增 `use_dynamic_bins` / `num_dynamic_bins`（`# [P2-DBDU]` 块）。`build()` 用 `cfg.get(..., default)` 读取，旧 yaml 与 `v0-baseline` tag 仍可加载。

## 5. Config 开关

```yaml
# [P2-DBDU] Dynamic Bins（第一增量）
use_dynamic_bins: False     # 默认 False → 精确回到 baseline 固定 LID 桶，逐位不变
num_dynamic_bins: 80        # 默认与 num_depth_bins 一致，分类头通道不变
```

## 6. 消融阶梯（规划）

| 行 | 配置 | 含义 |
|---|---|---|
| r1 | `use_dynamic_bins=False` | Baseline（固定 LID） |
| r2 | `use_dynamic_bins=True` | + Dynamic Bins（本增量最终行） |

后续增量在 r2 之上再叠 ADPM 残差头 / 几何深度融合 / L_ud 不确定性损失，各自独立消融。

## 7. 安全网（与 P1 同思路）

- **开关关 = baseline 原样**：`use_dynamic_bins=False` 时，前向走固定 `depth_bin_values`、损失走 LID 闭式，逐位不变，baseline 可原样复现。
- **分类头维度不变**：`depth_classifier` 仍输出 `num_bins+1` 通道，深度位置编码 `depth_pos_embed`（吃连续深度值）不受影响。
- **下游不动**：`weighted_depth` 经 `interpolate_depth_embed` 与 forward 里的 `grid_sample`(depth_map) 消费，二者只吃连续深度值，动态桶对它们透明。

## 8. 本增量明确不做（deferred）

ADPM 残差精修头 · 几何深度融合（注：MonoDETR 的 `depth_ave` 已含 `depth_geo` 几何项，后续增量在此基础上做 σ 加权）· 3D 位置编码 log-z 压缩 · L_ud 不确定性回归损失 · Bins 单调约束损失（已被 softmax+cumsum 构造取代）。

## 9. 借鉴文献

AdaBins (Bhat+CVPR2021，自适应深度桶) · 于承峄 MonoDBDU（动态 Bins + 不确定性融合，本方向直接来源） · GUPNet（深度不确定性 + 多次运行 mean±std 的评测范式） · 杨帅兵 ADPM（残差精修，后续增量） · 苏卫星 MVPI（几何深度与回归深度的 σ 加权融合，后续增量）。

## 评测口径（沿用 P1 教训）

- 报告指标 `AP_R40 3D`，重点看 **Ped/Cyc 的 Mod 和 Hard** 列（小目标叙事），不只看 Car Mod。
- `checkpoint_best` 只按 **Car Mod** 选帧，run 间会抖、偶尔撞 pre-decay 帧（见 P1 经验）；对照请用**自复现 baseline 的多次 run mean±std**，不要拿论文 val 20.61 单点当对照。

## 10. 第一增量复盘：朴素动态桶**塌缩** → 有界约束修复（2026/06/09）

### 10.1 r2 负结果

`use_dynamic_bins=True`（自由 softmax 版）训练一次（epoch 164）：

| 类别 Mod | 动态桶 r2 | False 对照（同代码同 seed） | Δ |
|---|---|---|---|
| Car | 19.23 | 19.44 | **-0.21** |
| Pedestrian | 5.58 | 6.41 | **-0.83** |
| Cyclist | 4.31 | 4.78 | **-0.47** |

三类 Mod 全部小跌（幅度在噪声带内，但无任何正信号）。

### 10.2 诊断（[diagnose_dynamic_bins.py](diagnose_dynamic_bins.py)，两层）

- **Tier 1（权重）**：末层 linear Frobenius norm = **2.77**（init=0）→ 桶**不是冻结**在 LID，确实对图像内容有响应。
- **Tier 2（真实 val 图的 per-image edges）**：表面"激活"，实则**塌缩**：
  - 跨图 std：max **0.31m**、mean **0.05m** → 16 张图的桶**几乎一样**，根本不是"per-image"，更像"一套学坏了的全局桶"。
  - 偏离 LID：mean **30.89m**、max **52.91m** → 塌到离 LID 极远的退化点。
  - 边界塌缩形态（vs LID 参考）：idx10 ≈ **20m**（LID 1.02）、idx20 ≈ **56m**（LID 3.89）、idx40 ≈ **59.8m**（LID 15.19）→ **80 桶里约 60 个被挤死在 [56,60]m 薄片**，真正有用的 0~56m 只剩 ~20 个有效桶。

**根因**：`softmax + cumsum` 自由参数化给了优化器"自我阉割分辨率"的自由度——它把概率全压在前 ~20 桶（中心已铺满 0~56m，`weighted_depth` 照样算对），剩 60 桶直接扔到 60m 边界。LID 因为写死，强制 80 桶全程有效，反而更好。这就是 r2 略劣于 baseline 的机制。

> ⚠️ 脚本末尾的自动 verdict 阈值设得过粗（只看 std>0.05 就判"genuinely per-image"），会误导；真正的信号是 **drift（塌缩）**，以人工读数为准。

### 10.3 修复：LID 先验 + tanh 有界扰动

把自由 softmax 换成**「固定 LID log 宽度先验 + 逐图有界扰动」**（[depth_predictor.py](lib/models/monodetr/depth_predictor/depth_predictor.py) `__init__` 与 `predict_dynamic_bins`）：

```python
delta  = torch.tanh(self.bin_predictor(global_feat)) * self.bins_adapt_scale   # |delta| <= scale
widths = F.softmax(self.lid_log_widths + delta, dim=-1) * (depth_max - depth_min)
```

- 末层 **zero weight + zero bias**，`delta=0`（iter 0）→ 精确复现 LID，平滑起步不变；
- `tanh×scale` 把每个 logit 的扰动钳在 ±scale，renorm 后每桶宽锁在 LID 的约 `e^±scale` 倍内（`scale=0.5` → 0.6×~1.65×），**60 桶再也塌不到边界**；
- 逐图自适应保留，仅幅度有界。
- 新增 config：`dynamic_bins_adapt_scale: 0.5`（[configs/monodetr.yaml](configs/monodetr.yaml)）。

### 10.4 验证口径

重训后**先跑 diagnostic 再看 AP**：Tier-2 的 `mean |edge - LID|` 应从 **30.9m 大幅降到几米内**；AP 预期**回到/接近 baseline**（跨图 std 仅 0.05m，说明 KITTI 全局深度分布逐图本就接近，单靠 per-image 桶涨点空间有限）。本增量的论文价值主要是坐实"**朴素动态桶会塌缩 → LID 有界约束修复**"这段干净分析；真正涨点留给后续碰 `depth_ave` 主路的增量（几何深度 σ 加权 / ADPM 残差）。

### 10.5 有界版重训复盘：塌缩已修，但 per-image 也被 tanh 饱和压没（2026/06/10）

有界版（`dynamic_bins_adapt_scale=0.5`）重训一次（best **20.09 @ epoch 158**），跑 diagnostic：

**Tier 1**：末层 Frobenius = **1.82**（init 0）→ 脚本判"对图像有响应"。但 `bin_predictor.2.bias` drift **mean 4.66 / max 8.10**。

**Tier 2**：跨图 std = **0.0000m**（6 张图逐行完全一致）；drift from LID mean **6.13m** / max **14.47m**。

→ 塌缩**确实修好了**（drift 从 30.9m 降到 6.1m，60 桶不再挤进远端 sliver），但 spread 不是 10.4 预期的"被限幅在小范围"，而是**精确塌到 0——比塌缩版（std 0.05m）还更不 per-image**。

**新根因（10.4 没预料到）：tanh 饱和**。`delta = tanh(bin_predictor(global_feat)) * scale`，bias 也在 tanh 内；bias drift 涨到 4.66（`tanh(4.66)≈0.9998`），把 tanh 顶进饱和区，`tanh'≈4e-4`，逐图项 `W2·relu(...)` 的梯度被掐死 → 每图 `delta≈±scale` 饱和值、partition 完全一致。**Tier-1 的非零 weight 是红鲱鱼**：有值但被饱和压没了表达力。净结果 = 一个**全局重调的静态 partition**，不是 dynamic。

**AP（vs 三次自复现 baseline，AP_R40 3D）**：

| | Car Mod | Ped Hard | Cyc Hard |
|---|---|---|---|
| baseline 区间 | 18.72–20.05 | 4.10–6.64 | 3.25–4.27 |
| 有界版 r2 | **20.09** | 4.74 | 3.76 |

三列全部落在 baseline 噪声带中段，**无任何可归因于 dynamic bins 的增益**；且 best=epoch158 ≠ 195，Ped/Cyc 报的是 Car-argmax 帧，本就不可靠。

### 10.6 决策：本增量收为"消融格"，转 depth-head 主路（2026/06/10）

两版（自由 softmax 塌缩 / tanh 有界饱和）合起来给出一个干净结论:**纯调 bin 边界这个杠杆对最终 AP 无效**——因为 bin_centers 只经 `weighted_depth → 深度位置编码 + 前景深度图 aux loss` 间接影响 decoder，最终 3D 框深度走的是独立的 per-query `depth_embed(mean, logσ)` 头，bins 够不到。

- **动作**：`use_dynamic_bins=False`（已设），dynamic bins 作为**消融证据保留**，不再深挖 per-image 机制。
- **论文用途**：用"朴素塌缩 → 有界约束 → per-image 仍被饱和压平 + AP 无增益"这段分析，论证**为什么必须叠加直接作用于 per-query 深度的组件**（不确定性 L_ud / 几何深度 σ 加权 / ADPM 残差），引出 P2 后续增量。
- **下一步**：转 §8 deferred 列表里碰 `depth_ave` 主路的部分（几何深度 σ 加权融合 / ADPM 残差精修 / L_ud），这些是强杠杆。

## 11. 增量 2b：几何深度 σ 加权融合（逆方差 + 解析 σ_geo，2026/06/10）

### 11.1 打的短板（承接 §10.6）

第一增量证明 bin 边界是弱杠杆。最终 3D 框深度由 [monodetr.py:256](lib/models/monodetr/monodetr.py#L256) 的 `depth_ave` 产出，那里三个深度源——回归 `depth_reg`、几何 `depth_geo = f·H/h`、深度图 `depth_map`——是**朴素 `/3` 平均**，σ 只有回归头那一个。`/3` 对所有 query 一视同仁：远处/小框时几何深度（依赖 box 高 h，h 小→噪声大）和回归深度都不可靠，却仍被等权平均。

### 11.2 核心做法：三源逆方差（精度）加权融合

精度 `p_i = 1/σ_i² = exp(-lv_i)`，按精度加权，**复用现有 σ 框架、不新增模块/损失**（仅 2 个标量参数）：

- **回归**：`lv_reg = depth_reg[...,1]`（现有学习头，逐 query）。
- **几何**：给一个**解析不确定性**。由 `D_geo = f·H/h` 误差传播 `σ_geo = D_geo·σ_h/h` ⇒ `lv_geo = 2·(log D_geo + log σ_h − log h)`，`σ_h = exp(geo_log_sigma_h)` 是**一个可学标量**（box 高像素不确定性）。**远处/小框 h 小 → σ_geo 大 → 几何自动降权**；近处大框 → 几何升权。
- **深度图**：`lv_map = depthmap_log_var`（**一个可学标量**，全局）。

```text
fused    = (p_reg·d_reg + p_geo·d_geo + p_map·d_map) / (p_reg+p_geo+p_map)
lv_fused = -log(p_reg+p_geo+p_map)        # 合并 log-variance → 回填 loss slot
depth_ave = cat([fused, lv_fused], -1)    # 仍 2 通道
```

loss 端 `exp(-lv_fused) = ΣP`，自动变成精度感知 NLL，**[loss_depths](lib/models/monodetr/monodetr.py#L394) 一行不动**。三精度各 clamp 到 `[e^-10, e^10]`，`P`/`lv_fused` 有界，不会 NaN。

### 11.3 改动清单（3 处，全在已知文件）

| 文件 | 改动 |
|---|---|
| [configs/monodetr.yaml](configs/monodetr.yaml) | 新增 `use_geo_depth_fusion`（默认 False）/ `geo_depth_init_pixel_sigma`（默认 1.0），`# [P2-DBDU]` 块 |
| [monodetr.py](lib/models/monodetr/monodetr.py) `__init__`/`build` | kwarg + `cfg.get` 透传；on 时注册 `geo_log_sigma_h` / `depthmap_log_var` 两个标量 `nn.Parameter`（全局，不进 `_get_clones`） |
| [monodetr.py](lib/models/monodetr/monodetr.py) `forward`（256–257） | `if use_geo_depth_fusion:` 逆方差融合；`else:` 原样 `/3`（字节级 baseline） |

### 11.4 安全网 / 旋钮

- **off = baseline 原样**：`use_geo_depth_fusion=False` 时 else 分支逐位不变。
- **on-init 有界**：clamp 保证不爆炸；on-init 不精确等于 `/3`（lv_reg 来自网络无法对齐），真正 safety 是 off 开关。
- **早期几何噪声**：训练初期 `size3d` 随机 → `d_geo` 不准。缓解旋钮 = 调大 `geo_depth_init_pixel_sigma`（初始全局降权几何），`σ_h` 可学会自适应。

### 11.5 消融阶梯

| 行 | 配置 | 含义 |
|---|---|---|
| r1 | `use_geo_depth_fusion=False` | Baseline（/3 平均） |
| r2 | `use_geo_depth_fusion=True` | + 几何 σ 加权融合（本增量） |

评测口径沿用 §「评测口径」：重点看 **Ped/Cyc 的 Mod/Hard**，固定 epoch 比对，勿只看 checkpoint_best 帧。

### 11.6 r2 负结果：灾难性崩溃 → 零初始化残差门修复（2026/06/11）

**首版 r2（`use_geo_depth_fusion=True`，无门）训练崩溃**：best Car Mod 3D **10.90 @ epoch 156**（baseline ~20），且 **2D bbox AP 也从 ~89 崩到 ~40**、**Car Easy < Mod 倒挂**、aos 全崩。不是噪声，是整个检测器坏了。

**根因（一因三症）**：`depth_geo = size3d_h / box2d_height · f` 同时耦合**预测的 3D 高 `size3d`** 和**预测的 2D 框高 `box2d_height = coord[:,4]+coord[:,5]`**（[monodetr.py:250-252](lib/models/monodetr/monodetr.py#L250)）。初始化时 `size3d` 小且随机 → `d_geo` 是垃圾。逆方差融合却对它双重过度信任:

| 机制 | 初始化时数值（近/高框） | 后果 |
|---|---|---|
| 点估计权重 `p_geo/P` | **0.999** | fused 深度≈垃圾几何 |
| loss 精度乘子 `exp(-lv_fused)` | **3602×** | 深度损失被放大数千倍 |

放大后的深度损失经 `depth_geo` 灌进 **2D 框高 + size3d** → 近(Easy)目标的 2D 框和 3D 尺寸被torch烂 → Easy<Mod 倒挂、bbox/aos/3d 全崩、无 NaN（持续坏平衡而非爆炸）。**正中交付时 flag 的 #4 风险**；原计划的 `σ_h` 旋钮太弱（init `size3d` 小 → `d_geo` 小 → `lv_geo` 被压负，σ_h 拉不回）。真正的缺陷:**计划缺 iter-0=baseline 保证**。

**修复:零初始化残差门**（沿用 P1 DCN / dynamic-bins 的零初始化安全网）。blend `baseline /3 mean` ⊕ `逆方差融合`，权重 `α = sigmoid(geo_fusion_gate)`，gate init −4 → α≈0.018 → **iter-0 ≈ 精确 baseline**（深度 + σ 都是）。几何只有在优化器主动调大 gate（即融合确实降 loss）时才获得权重 → **崩溃不可能**（下限=baseline），上行=学到的 σ 加权。

实测 init（最坏的近/高框）：几何权重 0.999→**0.345**(≈baseline 0.333)，loss 乘子 3602×→**1.2×**(≈baseline 1.0×)。

- 新增 config `geo_fusion_gate_init: -4.0`（[configs/monodetr.yaml](configs/monodetr.yaml)）；新增标量 `geo_fusion_gate`（[monodetr.py](lib/models/monodetr/monodetr.py) `__init__`）。
- forward 改为 gated-residual（[monodetr.py:265](lib/models/monodetr/monodetr.py#L265)）。
- **下次重训预期**:不再崩溃,下限回到 baseline ~20;若几何加权有用则 gate 自学打开、Ped/Cyc 上行。`gate_init` 调到 −2/−3 可让融合更早engage（更激进）。

### 11.7 门焊死 → clamp-linear 门 + loss-σ 解耦（公平测试，2026/06/12）

**11.6 的零初始化门跑出来 = baseline（Car Mod 19.76），诊断发现门焊死**（[diagnose_geo_fusion.py](diagnose_geo_fusion.py)）：

```text
geo_fusion_gate : -3.9920 (init -4.0) -> alpha 0.0181   gate moved +0.008
```

154 epoch 门只爬 +0.008。根因:**sigmoid 在 −4 处梯度 = α(1−α) = 0.018,被饱和压死**。σ_h（1.0→1.39）、depthmap_log_var（→−0.26）都动了，唯独门没动。→ **融合从未 engage,这次只是又跑了一遍 baseline,没真正测到几何加权**（不是融合没用，是没测到）。

**修复(针对"焊死",做公平测试)**:

1. **门换 clamp-linear**:`alpha = clamp(geo_fusion_gate, 0, 1)`,init 0.05。梯度恒为 1（不饱和）→ 门能自由移动；init α=0.05 → 点估计几何权重 0.367≈baseline 0.333，安全。
2. **loss-σ 与门解耦**:forward 的 σ 通道直接喂 `depth_reg[...,1]`（baseline 学习头），**不喂**融合 `lv_fused`。→ **永久移除 11.6 那个 init 3602× 放大机制**;门开多大都不放大 loss，几何只改进点估计。loss 路径与 baseline 逐位相同。更贴 CLAUDE.md「复用现有 σ 头」。

实测 init（最坏近/高框）:几何权重 0.367≈baseline，loss 乘子恒 **1.0×**，box 耦合 **1.10×** baseline（崩溃版是 ~160×），门梯度 **1.0**（焊死版 0.018）。**安全且能 engage**。

- config 改 `geo_fusion_gate_init: 0.05`;新增标量逻辑同前。诊断脚本解读已翻转:**clamp-linear 下"门停在 0.05"= 优化器试过、开门不降 loss = 几何加权中性（真负结果，可收口）**;"门爬过 0.2"= engage 了，看 AP。
- **判读**:重训后先跑 diagnose_geo_fusion.py。门爬起来 + Ped/Cyc 涨 → 有用;门爬起来但 AP 平 / 门停 0.05 → 干净负结果，收口转 ADPM 残差。

### 11.8 clamp-linear 版收口：固定 epoch195 对齐噪声后 = 干净负结果（2026/06/16）

clamp-linear 版训练完（`Best 19.318 @ epoch150`）。`checkpoint_best`（按 Car Mod 选帧）一度报 Ped Mod +1.62 / Ped Hard +1.13，看似有戏；但那是**选帧噪声**，不是几何融合的增益。

**关键证据：baseline 自己换 seed 的 epoch195 抖动幅度，已盖过这个"涨点"。** 三个自复现 baseline 的 epoch195：

| AP_R40 3D (epoch195) | base run1 | base run2 | base run3 | base 区间 | 几何融合 clamp-linear |
|---|---|---|---|---|---|
| Car Mod | 18.17 | 19.12 | 19.34 | 18.2–19.3 | 18.94 |
| **Ped Mod** | 5.63 | 7.87 | 6.48 | 5.6–7.9 | 7.29 |
| **Ped Hard** | 4.47 | 6.41 | 5.27 | 4.5–6.4 | 5.91 |
| **Cyc Mod** | 4.15 | 4.60 | 3.35 | 3.4–4.6 | 4.02 |
| **Cyc Hard** | 4.11 | 4.34 | 3.28 | 3.3–4.3 | 3.52 |

**固定 epoch195，几何融合六列全部落在 baseline 三 seed 噪声带正中，无一列突破上沿。** 光是 baseline 换 seed，Ped Hard 就在 4.47–6.41 之间晃近 2 个点；几何融合的 5.91 正好坐在中间。→ checkpoint_best 报的 Ped +1.6 是抽样噪声，**几何融合 σ 加权融合对 AP 无可归因增益**。

**决策：几何融合 clamp-linear 版收口为干净负结果。** 与 §10.6 的 dynamic bins 合起来，P2 两个"复用现有 σ 框架 / 不引入新模块"的轻量改法（调 bin 边界、调三源融合权重）都**对最终 AP 中性**。机制层面的共同原因：二者都只能间接影响最终 3D 框深度，而最终深度由 per-query `depth_embed` 头主导——**要涨点必须上一个直接、且带新表达力的深度组件**（ADPM 残差头 / 新的不确定性损失 / 借鉴最新论文的深度模块），不能再靠重新加权已有的源。

> ⚠️ 评测方法论教训（已坐实）：本项目 `checkpoint_best` 按 Car Mod 选帧，Ped/Cyc 报的是 argmax-Car 帧的顺带值，**单次 run 的 Ped Hard 噪声实测达 ±2 AP**（比之前估的 ±1.3 还大）。今后任何 P2/P3 改动判增益，**必须固定 epoch195 且对齐 baseline 多 seed 区间**，禁止用 checkpoint_best 的 Ped/Cyc 单点下结论。
