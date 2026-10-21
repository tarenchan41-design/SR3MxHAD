# 任务：SR3M × HAD 项目架构统一部署

你是我在 RTX 3070 笔记本（Windows + WSL）上的本地开发 agent。请按本文档分阶段完成项目架构的统一。**每个阶段结束都有明确的停止点，到停止点必须停下来向我汇报，等我确认后再继续。**

---

## 0. 背景（请先完整读完）

### 0.1 研究目标
- **SR3M**：自建的无人机视觉语言导航（VLN）仿真数据集，以复杂地形为特色。场景在 Blender 中建模后导入 UE 5.8（本机实际版本 5.8.4）。共 8 类场景 × 4 张地图 × 约 50 条轨迹，约 1,606 条，指令为英文，`instruction_zh` 字段保留中文。**计划之后公开发布到 Hugging Face。**
- **HAD**：前视 + 俯视双视角融合的导航模型（ResNet50 + Embedding/BiLSTM），之前在 OpenUAV 上做过离线实验。
- **SR3MxHAD**：用 HAD 模型在 SR3M 上训练、推理、离线评测，并通过 Cosys-AirSim 3.5.0 做闭环仿真。论文目标是 IEEE RA-L（IROS 2027 选项），**11/3 前要完成第一轮离线对比实验**。本次架构整理是为它服务的基础设施，不能拖慢它。

### 0.2 现有代码和数据分布

| 内容 | 位置 | 备注 |
|---|---|---|
| HAD 代码（副本 1） | H3C 电脑：`C:\Users\24333\Desktop\university\小论文\双视角融合idea` | **你无法直接访问**，需要我在那台机器上操作 |
| HAD 代码（副本 2） | 本机 WSL：`/home/qlj/h3c_pro` | 跑过实验，改过 |
| HAD 代码（副本 3） | GitHub：`https://github.com/qlj215/HAD-UAV-VLN` | 之前多次提交；整合文档记录的来源为 main 分支 commit `f01440a` |
| SR3M 生产代码 | 本机 Windows：`E:\DL\Transformers\Projects\SR3M` | Blender / UE5 / 规划器 / 渲染相关；不确定是否已纳入 git |
| SR3M 数据集 | 本机：`F:\WSL_ColdStorage\datasets\DatasetCaptureBatches\workbench` | 是**可变的工作区**，不能直接用于训练 |
| SR3MxHAD 代码 | 云服务器（RTX 5090，Linux） | 目前主要在云端编写；云端**不能跑仿真** |
| SR3MxHAD 计划目录 | 本机：`E:\DL\Transformers\Projects\SR3MxHAD` | 将作为本机的工作副本 |

**关键问题**：HAD 的三份副本以及云端的 SR3MxHAD，**我自己也不清楚哪份最新**。本机和云端都跑过实验、都改过代码。

### 0.3 机器分工（目标状态）
- **本机 RTX 3070**：数据生产（Blender、UE5）+ 闭环仿真（需要 UE）+ 少量本地训练
- **云端 RTX 5090**：训练、离线评测
- **H3C 电脑**：只用于写论文，代码副本冻结

### 0.4 目标架构

```
GitHub
├── HAD-UAV-VLN        → 对账后打标签冻结（openuav-final），只读归档
├── SR3MxHAD           → 唯一的活跃代码仓库（私有，暂不公开）
└── SR3M（生产代码）    → 纳入 git，只含脚本，不含大型资产

SR3MxHAD/
├── pyproject.toml
├── src/
│   ├── sr3m/          # benchmark：数据读取、划分、动作合同、坐标转换（UE ↔ NED）、
│   │                  #   评测指标（SR20 等）、Cosys-AirSim 闭环执行器
│   └── had/           # method：模型模块、词表、STOP 标签协议、损失函数
├── experiments/       # 唯一同时引用 sr3m 和 had 的地方：配置、prepare/train/predict/evaluate/rollout 入口
├── configs/machines/  # 每台机器的路径配置：*.example.yaml 提交到 git，实际配置加入 gitignore
├── tests/             # 包含边界测试：src/sr3m 不得 import had
└── docs/
    ├── ARCHITECTURE.md        # 本次整理的结果说明
    ├── PROVENANCE.md          # had/ 中每个模块的来源（仓库、commit、修改说明）
    ├── DATA_RELEASES.md       # 数据发布版本记录
    ├── runs_registry.csv      # 实验登记表
    └── DATASET_CARD_draft.md  # Hugging Face 数据集卡片草稿（只起草，不发布）

数据
<数据根目录>/SR3M/releases/v1.0/
├── （数据文件）
├── manifest.json      # 每个文件的相对路径、大小、sha256；按场景/地图/划分的统计
├── splits_v1.0.json   # 当前使用的划分
├── splits_legacy.json # 原划分（1027/184/199/196），用于复现旧结果
├── schema.md
└── RELEASE_NOTES.md
```

**边界规则（因为 SR3M 要公开）**：`sr3m/` 是将来会单独公开的 benchmark 代码，**绝不能引用 `had/` 或 `experiments/`**。动作合同、坐标转换、成功判定（SR20）属于 `sr3m/`；STOP 正样本的标注规则（20 米末尾连续段）属于训练选择，放在 `had/`。

---

## 1. 安全规则（全程必须遵守）

1. **不删除、不移动、不覆盖**以下内容：workbench 数据目录、UE / Blender 工程文件、旧实验输出、checkpoint、旧仓库的工作目录。需要用到时一律**复制**。
2. **不 force push，不改写 git 历史，不直接提交到 main。** 所有改动在新分支上进行（例如 `arch/restructure`），由我审核后再合并。
3. **不提交大文件和敏感信息**：数据、图像、checkpoint、UE 构建产物、任何 token / 密钥 / SSH 配置。`git add` 之前先检查是否有超过 20 MB 的文件，有的话停下来问我。
4. **不上传到 Hugging Face，不把任何仓库设为公开。**
5. **不修改模型和训练逻辑。** 本次只做移动、重命名和修正 import。重构前后行为必须一致，用冒烟测试验证。
6. **不启动正式训练，不占用 GPU 跑长任务。**
7. 路径里有中文和空格，所有命令都要正确加引号。同一个仓库同时被 Windows git 和 WSL git 操作时，注意换行符和文件权限造成的假改动：设置 `core.autocrlf` 和 `core.filemode false`，并添加 `.gitattributes`。
8. 遇到本文档没有覆盖、且不可逆或有歧义的情况，**停下来问我**，不要自己猜。

---

## 2. 阶段一：只读盘点（不做任何修改）

对你能访问的每个位置（WSL 中的 `/home/qlj/h3c_pro`、`E:\...\SR3M`、`E:\...\SR3MxHAD`、GitHub 上的 HAD-UAV-VLN、workbench 数据目录）记录：

- 是否是 git 仓库；当前分支；最近 10 条提交；是否有**未提交的改动、未推送的提交、stash**；remote 地址
- 目录结构概要（两层）；主要入口脚本
- 是否有大文件（超过 20 MB）以及 `.gitignore` 是否覆盖了它们
- 数据目录：总大小、文件数、按场景 / 地图的统计；划分文件在哪里、四个划分各自叫什么名字；旧 desert 场景的 episode 数量及分布在哪些划分里
- 各个盘的剩余空间（后面复制数据需要）

把结果写入 `E:\DL\Transformers\Projects\SR3MxHAD\docs\INVENTORY_<日期>.md`。这是你在本阶段唯一允许创建的文件；如果目录还不存在，可以只为这个文件创建它。

### 停止点 1
汇报盘点结果，并告诉我需要在 **H3C 电脑**和**云服务器**上执行的快照命令（参考附录 A，根据你的盘点结果调整）。等我执行完成后再继续。

---

## 3. 阶段二：对账（找出哪份代码最新）

做法：**把每一份副本都推到 GitHub 上各自独立的快照分支**，然后集中比较。这样不会覆盖任何一份，也能看清全部差异。

1. 本机 WSL 副本：按附录 A 的方法创建 `snapshot/wsl3070-<日期>` 分支，把所有未提交的改动提交进去并推送（先做大文件检查）。
2. 等我完成 H3C 和云端的快照（分支 `snapshot/h3c-<日期>`、`snapshot/cloud5090-<日期>`；云端的 SR3MxHAD 同理，推送到 SR3MxHAD 仓库）。
3. 抓取所有快照分支，比较：
   - 各分支相对 `main` 和 `f01440a` 的提交与文件差异
   - 分支之间的差异：哪些文件只在某一份里改过，哪些文件**多份都改过且内容冲突**
   - 对每个有差异的模块，用几句话概括改了什么，例如"增加了 STOP 头"或"修改了数据读取"
   - 重点检查：SR3MxHAD 中实际使用的 HAD 模型模块，与各份副本里的对应文件是否一致
4. 写出 `docs/RECONCILIATION_REPORT.md`，内容包括：
   - 差异总表（文件 × 副本）
   - 你建议的权威版本，以及理由
   - 需要我拍板的冲突列表：**不要自动合并冲突的逻辑**

### 停止点 2
汇报对账结果，等我确认权威版本和冲突的处理方式。

确认之后：在 HAD-UAV-VLN 上合并出一个 `reconciled` 提交，打标签 `openuav-final`，在 README 顶部注明"已归档，后续开发转移到 SR3MxHAD"。**本地的旧副本目录保持原样，不删除。**

---

## 4. 阶段三：建立 SR3MxHAD 仓库结构

1. 如果 SR3MxHAD 还没有 GitHub 仓库，**先问我**，确认后再创建私有仓库。以云端快照为基础，在本机 clone 到 `E:\DL\Transformers\Projects\SR3MxHAD`。
2. 在分支 `arch/restructure` 上按 0.4 的结构重组：
   - 现有代码按职责拆分到 `src/sr3m/`、`src/had/`、`experiments/`
   - `had/` 中来自 HAD 的模块，在 `docs/PROVENANCE.md` 里记录来源（仓库、`openuav-final` 的 commit、相对原版的改动）
   - 保留并统一 `prepare / train / predict / evaluate / rollout` 五个入口
   - 消除对相邻目录的 `sys.path` 依赖；改为 `pip install -e .`
   - 所有路径从 `configs/machines/<机器名>.yaml` 读取，代码里不出现硬编码路径。提供三个示例配置：`rtx3070_win.example.yaml`、`rtx3070_wsl.example.yaml`、`cloud5090.example.yaml`，每个都包含数据根目录、数据版本、输出根目录、UE 工程路径和 Cosys-AirSim 路径
3. 添加测试：
   - **边界测试**：扫描 `src/sr3m/` 下所有 import，出现 `had` 或 `experiments` 即失败
   - **坐标测试**：用一条 mountain 爬升轨迹验证 dz > 0，转换到 NED 后为负；覆盖 yaw 跨 ±180° 的情况。如果坐标转换代码还不存在，先留一个标注清楚的待办，不要自己实现新逻辑
   - **冒烟测试**：用 2–3 个 episode 跑通 prepare，再跑一个训练 step，确认重构前后输出一致
4. `.gitignore` 覆盖：数据、输出、checkpoint、`configs/machines/*.yaml`（不包括 `*.example.yaml`）、UE 中间文件、`.env`。

### 停止点 3
汇报改动摘要和测试结果，由我审核分支，合并到 main 由我决定。

---

## 5. 阶段四：冻结数据发布版 v1.0

1. 根据阶段一统计的数据大小和剩余空间，提出 `releases/v1.0` 的存放位置，**等我确认**。
2. 从 workbench **复制**（不是移动）到发布目录，生成：
   - `manifest.json`：每个文件的相对路径、大小、sha256，以及按场景 / 地图 / 划分的统计
   - `splits_legacy.json`：原样保留原划分
   - `splits_v1.0.json`：**把旧 desert 场景的全部 episode 从所有划分中移除**，标记为 `excluded_deprecated`（desert 之后会在 Blender 中重建，作为未见地形测试集在 v1.1 中加入）。汇报新划分的数量变化
   - `schema.md`：字段说明、动作合同（当前 yaw 对齐的前 / 右 / 上，单位米；偏航增量单位弧度）、坐标系约定
   - `RELEASE_NOTES.md`
3. 写一个校验脚本 `sr3m/tools/verify_release.py`，根据 manifest 校验任意一台机器上的数据副本，供我在云端运行。
4. 把发布目录设为只读，并在 `docs/DATA_RELEASES.md` 中记录。

### 停止点 4
汇报发布结果和新的划分数量。

---

## 6. 阶段五：实验登记、备份方案、发布准备

1. `docs/runs_registry.csv`，字段：`run_id, date, machine, code_commit, data_release, split_file, config_path, seed, model_variant, status, key_metrics, checkpoint_path, notes`。把阶段一发现的旧中文实验登记为 `legacy`。再提供一个小脚本，让每次训练结束时自动追加一行。
2. **备份方案（只提出方案，不执行）**：列出需要备份的资产，包括 UE / Blender 工程、`releases/`、关键 checkpoint，以及各自的大小，给出建议的备份目标和频率。
3. `docs/DATASET_CARD_draft.md`：按 Hugging Face 数据集卡片的格式起草，包括数据来源、场景、划分、字段和已知局限。许可证一栏留空，由我决定。
4. 如果 `E:\...\SR3M` 生产代码还不是 git 仓库：初始化一个，配好 `.gitignore`，确保不包含任何大型资产，并汇报结构，由我决定是否推送到 GitHub。

### 停止点 5（最终）
把 `docs/ARCHITECTURE.md` 写完整，内容包括：做了什么、现在每样东西在哪里、各台机器的使用方式（如何拉取代码、如何指定数据版本）、尚未解决的问题。然后汇报。

---

## 7. 不在本次范围内（不要做）
- 把 `sr3m/` 拆成独立仓库（等公开发布时再做）
- 实现新的模型、STOP 协议或闭环逻辑
- Unreal MCP 相关配置
- 跑正式训练或闭环实验
- 修改 workbench 中的任何数据

---

## 附录 A：快照命令参考（在每份副本所在的机器上执行）

**如果该目录是 git 仓库：**
```bash
cd "<副本目录>"
git status --porcelain          # 查看未提交的改动
git stash list                  # 查看是否有 stash
git log --oneline -10
git branch -a
# 检查大文件（超过 20 MB）；有的话先停下，不要提交
find . -path ./.git -prune -o -type f -size +20M -print     # Linux / WSL
# Windows PowerShell:
# Get-ChildItem -Recurse -File | Where-Object { $_.Length -gt 20MB -and $_.FullName -notmatch '\\.git\\' }

git switch -c snapshot/<机器名>-<YYYYMMDD>
git add -A
git commit -m "snapshot: <机器名> working tree <YYYYMMDD>"
git push origin snapshot/<机器名>-<YYYYMMDD>
```
如果有 stash，用 `git stash show -p stash@{N} > stash_N.patch` 导出补丁，一并提交到快照分支。

**如果该目录不是 git 仓库**（H3C 上的文件夹可能就是这种情况）：
把整个文件夹（排除数据和 checkpoint）复制到本机 `E:\staging\had_h3c_snapshot_<日期>\`，由本机 agent 与仓库做对比。
