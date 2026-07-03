# MonoDETR 毕业论文 Git 版本管理指南

> 适用场景：以 MonoDETR 为 baseline、需要做 3 个改进方向（SCA-FPN / 动态 Bins / SMCA）+ 各方向独立消融 + 最终整合论文模型的研究项目。
> 目标：让你能在 baseline / +P1 / +P2 / +P3 / +全部 五种状态之间无缝切换、可重现、不污染。

---

## 1. 整体策略

### 1.1 分支结构（最终）

```
main          ─●──────────────────────────────────────────  ← 永远是 baseline，不动
                │   (tag: v0-baseline)
                ├──● ●──●──●─────────                       p1-sca-fpn       (方向一)
                │
                ├──────── ●──●──●──●────                    p2-dynamic-bins  (方向二，从 v0 派生)
                │
                ├──────────────── ●──●──●───                p3-smca          (方向三，从 v0 派生)
                │
                └──────────────────────────── ●──●──●──     thesis-final     (P1+P2+P3 整合分支)
                                              ↑
                                       合并三个方向 + 解冲突
                                       打 tag: v1.0-thesis-final
```

### 1.2 为什么这样做

| 设计 | 理由 |
|---|---|
| **main 永远停在 baseline** | 毕业论文需要 baseline 作为对照组，必须随时可重现 |
| **每个方向单独开分支** | 实验出问题时只影响一个方向；论文章节差异 `git diff main..p1` 一目了然 |
| **每个方向都从 baseline 派生**（不是从前一个方向） | 保证三个方向**独立可消融**，否则 P2 会自动带着 P1 的改动 |
| **最后开 thesis-final 整合** | 三个方向单独验证后再叠加，便于排查叠加冲突 |
| **用 tag 锚定关键节点** | tag 是不可变指针，比 commit hash 更直观（v0-baseline、p1-ablation-done 等） |

---

## 2. 现在立刻要做的事（首次设置，5 条命令）

**前提**：当前在 main 分支，本地工作区已经有方向一的 SCA-FPN 改动但还没提交。

```bash
cd /desay120T/ct/dev/uid01955/MonoDETR-main   # 或者你本地的工作目录

# 1) 确认当前状态：在 main 上，且有未提交的方向一改动
git status
git branch
git log --oneline -5
```

```bash
# 2) 给 baseline 打永久 tag（毕业论文的"对照组锚点"）
#    先 stash 一下未提交的改动，避免误把方向一代码打进 v0-baseline
git stash                                       # 把方向一改动暂存
git checkout main
git tag -a v0-baseline -m "MonoDETR baseline reproduced on KITTI val: AP3D Mod 16.47"
git push origin v0-baseline                     # 把 tag 推到 GitHub
```

```bash
# 3) 从 main 开出方向一分支
git checkout -b p1-sca-fpn
git stash pop                                   # 把方向一改动恢复出来
```

```bash
# 4) 提交方向一的所有改动
git add lib/models/monodetr/sca_fpn.py \
        lib/models/monodetr/monodetr.py \
        configs/monodetr.yaml \
        P1_SCA_FPN_Summary.md

git commit -m "feat(p1): add SCA-FPN cross-scale feature pyramid

- new module lib/models/monodetr/sca_fpn.py (CA + SA + top-down guidance)
- monodetr.py: wire SCA-FPN between input_proj and depth_predictor
- configs/monodetr.yaml: add use_sca_fpn toggle (default True)
- P1_SCA_FPN_Summary.md: thesis chapter draft

Expected gain: +1.5~2.5 pp on Car Mod AP3D for small/distant targets."
```

```bash
# 5) 推到 GitHub
git push -u origin p1-sca-fpn
```

执行完之后 GitHub 仓库长这样：

```
yourrepo
├── main            (= v0-baseline tag, 是 baseline)
└── p1-sca-fpn      (方向一开发分支)
```

✅ main 还是干净的 baseline，✅ tag `v0-baseline` 永久指向它，✅ 方向一改动隔离在 `p1-sca-fpn` 分支。

---

## 3. 在方向一分支上的日常迭代

### 3.1 调超参 / 小改动

```bash
# 确保在 p1-sca-fpn 分支
git checkout p1-sca-fpn

# 改完代码（比如 reduction 改成 8）
git add configs/monodetr.yaml
git commit -m "tune(p1): reduce CA bottleneck from 16 to 8 (better for small obj)"
git push
```

### 3.2 阶段性里程碑打 tag

每次有阶段性结果（消融实验做完一组、训练完出 AP）就再打一个小 tag：

```bash
git tag -a p1-ablation-done -m "P1 ablation done: best Mod AP3D 17.92 with CA+SA+residual"
git push origin p1-ablation-done
```

### 3.3 提交规范（建议）

提交 message 用「类型(范围): 简述」格式：

| 类型 | 用途 | 示例 |
|---|---|---|
| `feat` | 新功能 | `feat(p1): add SCA-FPN` |
| `fix` | 修 bug | `fix(p1): wrong gn group num when channel=128` |
| `tune` | 调超参 | `tune(p1): try reduction=8` |
| `docs` | 文档 | `docs(p1): update result table` |
| `refactor` | 重构 | `refactor(p1): extract attention modules` |
| `exp` | 实验记录 | `exp(p1): ablation w/o residual, Mod 16.21` |

---

## 4. 方向二、方向三怎么开

**核心原则：每个方向都从 `v0-baseline` 派生，不要从前一个方向派生**。

### 4.1 方向二（动态 Bins + 深度不确定性）

```bash
# 先把方向一的工作 commit/push 干净，避免丢失
git checkout p1-sca-fpn
git status                    # 必须是 clean working tree

# 从 v0-baseline 出发开新分支
git checkout v0-baseline      # ← 注意是从 tag 出发，不是 p1-sca-fpn
git checkout -b p2-dynamic-bins

# 开发、commit、push（同方向一流程）
# ...
git push -u origin p2-dynamic-bins
```

### 4.2 方向三（SMCA + 不确定性 V 门控）

```bash
git checkout v0-baseline
git checkout -b p3-smca
# ...
git push -u origin p3-smca
```

### 4.3 三个方向并行（高级技巧）

如果想在不同终端 / 不同服务器同时开发三个方向，可以用 git worktree：

```bash
# 在 baseline 目录下创建 3 个并行工作树
git worktree add ../monodetr-p1 p1-sca-fpn
git worktree add ../monodetr-p2 p2-dynamic-bins
git worktree add ../monodetr-p3 p3-smca

# 之后 ../monodetr-p1 就是方向一分支的工作目录，互不干扰
```

---

## 5. 最终整合阶段（论文投稿前 1 个月）

三个方向都做完、各自消融跑完之后，开 `thesis-final` 分支整合：

```bash
git checkout main
git checkout -b thesis-final

# 依次合并三个方向
git merge p1-sca-fpn          # 通常无冲突
git merge p2-dynamic-bins     # 可能有冲突，主要在 monodetr.py / yaml
git merge p3-smca             # 可能有冲突
```

### 5.1 冲突主要在两处，提前预备

- `lib/models/monodetr/monodetr.py`：三个方向都改 `__init__` 和 `forward`
- `configs/monodetr.yaml`：三个方向都加配置字段

**减少冲突的写码约定**：
- 每个方向用统一注释前缀（`# [P1-SCA-FPN]` / `# [P2-DBDU]` / `# [P3-SMCA]`），合并时一目了然
- yaml 配置字段也按方向分块加注释分隔

### 5.2 解决冲突 → 提交 → 打最终 tag

```bash
# 用编辑器打开冲突文件，手动合并 P1+P2+P3 三段代码
# 解决后
git add lib/models/monodetr/monodetr.py configs/monodetr.yaml
git commit                    # 自动生成 merge commit message

git push -u origin thesis-final

# 打最终 tag
git tag -a v1.0-thesis-final -m "Final thesis model: SCA-FPN + DynBins + SMCA"
git push origin v1.0-thesis-final
```

---

## 6. 消融实验的分支切换示例

毕业论文最关键的消融表通常长这样：

| 实验 | 配置 | 操作 |
|---|---|---|
| ① Baseline | 原版 MonoDETR | `git checkout v0-baseline && bash train.sh` |
| ② +P1 only | + SCA-FPN | `git checkout p1-sca-fpn && bash train.sh` |
| ③ +P2 only | + 动态 Bins | `git checkout p2-dynamic-bins && bash train.sh` |
| ④ +P3 only | + SMCA | `git checkout p3-smca && bash train.sh` |
| ⑤ +P1+P2 | 临时组合 | 见 §6.1 |
| ⑥ +P1+P3 | 临时组合 | 见 §6.1 |
| ⑦ +P2+P3 | 临时组合 | 见 §6.1 |
| ⑧ 全部 | thesis-final | `git checkout thesis-final && bash train.sh` |

**每次切完分支必做：**
1. `git status` 确认 clean
2. 打开 [configs/monodetr.yaml](configs/monodetr.yaml) 确认配置项是预期的
3. 训练前在日志里记录当前 commit hash：`git rev-parse HEAD >> outputs/run_info.txt`

### 6.1 临时组合消融（不需要长期保留）

```bash
# 想跑 P1+P2 消融，但不想新建分支
git checkout p1-sca-fpn
git checkout -b tmp-p1p2          # 临时分支
git merge p2-dynamic-bins         # 合并 P2
# 解冲突 → 训练 → 出结果
# 结果回填到论文之后，删掉临时分支
git checkout thesis-final
git branch -D tmp-p1p2
```

---

## 7. .gitignore 必备项

仓库根目录 `.gitignore` 文件应该包含：

```gitignore
# === 训练产物 ===
outputs/
logs/
*.pth
*.pkl
*.bin

# === 数据集（不应该提交）===
data/
data/KITTIDataset/

# === Python 缓存 / 编译产物 ===
__pycache__/
*.pyc
*.pyo
*.so
build/
*.egg-info/

# === Deformable DETR ops 编译产物 ===
lib/models/monodetr/ops/build/
lib/models/monodetr/ops/dist/
lib/models/monodetr/ops/*.egg-info/

# === IDE / 系统 ===
.vscode/
.idea/
.DS_Store
Thumbs.db

# === 本地特定配置（可选）===
configs/*.local.yaml
configs/local_*.yaml

# === Jupyter / 临时 ===
.ipynb_checkpoints/
*.ipynb
tmp/
```

### 7.1 验证 .gitignore 生效

```bash
git status --ignored          # 查看所有被忽略的文件
git check-ignore -v outputs/  # 查某个具体路径为什么被忽略
```

### 7.2 如果已经误提交了大文件

```bash
# 例如不小心把 outputs/checkpoint.pth 提交了
git rm --cached outputs/checkpoint.pth     # 注意 --cached，不删本地文件
git commit -m "chore: remove accidentally committed checkpoint"

# 如果文件很大且历史上有多次提交，需要重写历史（小心，会改动 commit hash）
# 推荐用 git-filter-repo（比 filter-branch 快很多）：
# pip install git-filter-repo
# git filter-repo --path outputs/checkpoint.pth --invert-paths
```

---

## 8. 常用操作速查

### 8.1 查看当前状态

```bash
git branch                     # 当前分支 + 所有本地分支
git branch -a                  # + 远程分支
git tag -l                     # 所有 tag
git log --oneline -10          # 最近 10 个 commit
git log --all --oneline --graph -20  # 图形化全分支历史
```

### 8.2 切换、撤销

```bash
git checkout <branch>          # 切分支
git checkout <tag>             # 切到某个 tag（detached HEAD 状态）
git checkout -b <new> <base>   # 从 base 开新分支

git restore <file>             # 撤销工作区改动
git restore --staged <file>    # 取消暂存
git reset --hard HEAD          # 撤销所有未提交改动（危险）
git reset --hard v0-baseline   # 把当前分支回滚到 v0-baseline（危险，会丢失新 commit）
```

### 8.3 暂存当前改动切别的分支

```bash
git stash                      # 暂存
git stash list                 # 查看暂存列表
git stash pop                  # 恢复最新暂存
git stash apply stash@{1}      # 恢复指定暂存（不删除）
git stash drop stash@{0}       # 删除暂存
```

### 8.4 拉取 / 推送

```bash
git pull                       # 拉远端到当前分支
git pull --rebase              # 用 rebase 方式拉（线性历史）
git push                       # 推到远端跟踪分支
git push -u origin <branch>    # 首次推送 + 建立跟踪
git push origin --tags         # 推所有 tag
git push origin <tag>          # 推单个 tag
```

### 8.5 同步本地 / 远程

```bash
# 远程有新 commit，本地拉取
git fetch origin
git diff main origin/main      # 看远程比本地多了什么
git merge origin/main          # 合并远程到本地

# 删除本地某分支（远程的 p1-sca-fpn 不受影响）
git branch -d p1-sca-fpn       # 安全删除（未合并会报错）
git branch -D p1-sca-fpn       # 强制删除

# 删除远程分支
git push origin --delete p1-sca-fpn
```

### 8.6 看两个分支的差异

```bash
git diff main..p1-sca-fpn                    # 所有差异
git diff main..p1-sca-fpn -- lib/            # 只看 lib/ 下的差异
git diff main..p1-sca-fpn --stat             # 摘要（哪些文件改了多少行）
git log main..p1-sca-fpn --oneline           # 看 p1 比 main 多了哪些 commit
```

---

## 9. 常见坑 & 应对

### Q1: 不小心在 main 上改了代码怎么办？

```bash
# 还没 commit
git stash                      # 暂存
git checkout p1-sca-fpn        # 切到正确分支
git stash pop                  # 把改动恢复出来

# 已经 commit 但还没 push
git log --oneline -3           # 找到误提交的 commit hash
git checkout p1-sca-fpn
git cherry-pick <commit-hash>  # 把那个 commit 复制到 p1 分支
git checkout main
git reset --hard HEAD~1        # 撤销 main 上的误提交（危险，仅限未 push）

# 已经 push 了 → 别慌，从 main 把那个 commit 撤销掉
git checkout main
git revert <commit-hash>       # 生成一个反向 commit
git push
```

### Q2: 拉取时遇到冲突？

```bash
git pull
# CONFLICT (content): Merge conflict in xxx.py

# 1) 用编辑器打开冲突文件，找 <<<<<<< 标记手动合并
# 2) 标记解决
git add xxx.py
# 3) 完成合并
git commit
```

### Q3: 想看某个 tag 是哪个 commit？

```bash
git rev-list -n 1 v0-baseline
git show v0-baseline --stat
```

### Q4: 仓库太大想瘦身？

```bash
# 列出当前历史里最大的 10 个对象
git rev-list --objects --all \
  | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
  | sort -k3 -n | tail -10

# 看 .git 目录大小
du -sh .git
```

### Q5: 服务器和本地代码不同步？

```bash
# 服务器上你跑训练时改了 yaml 但忘了 commit
# 本地想拿到服务器的改动

# 服务器侧：
git add -A && git commit -m "exp: tuned on server"
git push

# 本地：
git pull
```

### Q6: 想看某次实验跑的是哪个版本？

提前在训练脚本里记录 commit hash：

```bash
# train.sh 开头加一行
git rev-parse HEAD > outputs/${EXP_NAME}/run_info.txt
git status >> outputs/${EXP_NAME}/run_info.txt

# 之后看 run_info.txt 就知道是哪次 commit
```

---

## 10. 完整工作流时间线（毕业论文 11 个月）

| 月份 | 阶段 | Git 操作 |
|---|---|---|
| 第 1 月 | baseline 复现 | 上传 baseline 到 GitHub main，打 tag `v0-baseline` |
| 第 2-4 月 | 方向一 SCA-FPN | 开 `p1-sca-fpn` 分支，完成开发 → tag `p1-final` |
| 第 5-7 月 | 方向二动态 Bins | 从 `v0-baseline` 开 `p2-dynamic-bins` → tag `p2-final` |
| 第 8-9 月 | 方向三 SMCA | 从 `v0-baseline` 开 `p3-smca` → tag `p3-final` |
| 第 10 月 | 整合 | 开 `thesis-final` 合并三方向 → tag `v1.0-thesis-final` |
| 第 11 月 | 写作答辩 | 论文用 `thesis-final` 截图代码；不再推主要新功能 |

### 10.1 答辩前最终检查清单

- [ ] `v0-baseline` tag 可重新跑出论文里 baseline 的数字
- [ ] `p1-final` / `p2-final` / `p3-final` 三个 tag 可重现各章节的消融数据
- [ ] `v1.0-thesis-final` 可重现摘要里的最终性能
- [ ] 所有 `outputs/*.pth` 大文件没污染仓库（用 `du -sh .git` 检查）
- [ ] README.md 有 baseline 复现命令、各方向训练命令、各 tag 对应的预期 AP
- [ ] 三个 `Pn_*_Summary.md` 文档完整，便于答辩问答

---

## 11. 一页纸速查（贴在显示器边上）

```
日常 4 步：
  git status                # 看状态
  git add <file>            # 暂存
  git commit -m "msg"       # 提交
  git push                  # 推送

切分支：
  git checkout <branch>
  git checkout -b <new>     # 新建并切

打 tag：
  git tag -a <name> -m "msg"
  git push origin <name>

撤销：
  git stash                 # 暂存改动
  git restore <file>        # 撤销未暂存改动
  git reset HEAD~1          # 撤销最新 commit（保留改动）

查看：
  git log --oneline -10
  git diff main..p1-sca-fpn
  git branch -a
```

---

*文档版本：2026-05-19 v1.0*
*配套阅读：`P1_SCA_FPN_Summary.md`（方向一改进文档）*
