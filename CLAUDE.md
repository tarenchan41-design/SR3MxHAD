# CLAUDE.md — SR3M × HAD

> 给 agent 的项目约定。发现与代码不符时以代码为准，并提醒我更新本文件。

## 1. 我是谁、怎么合作

- PhD 中期，研究方向：无人机视觉导航 / VLA。导师放养，研究基本自主推进。
- 早上精力最好；没有明确 deadline 时容易拖。请帮我把任务拆成**有明确产出的小步骤**，每次结束时告诉我"下一步是什么"。
- 算力：
  - 本机 RTX 3070 Laptop 8GB：数据生产、UE 渲染、闭环、少量训练。
  - 云端 RTX 5090：训练、抽特征、离线评测，不能跑仿真。由我手动 clone 并运行，agent 每次给出具体命令。
  - **优先低算力方案**，大规模训练前先跟我确认。
- 流水线按我的要求：**每个后续实验前先讨论**。
- 实验中发现的现象可以发展成新的论文点（不必都事先预注册），但要标明"事后发现、仅描述"，和预注册判据分开写。

## 2. 项目全貌

SR3M（以地形为核心的无人机 VLN 仿真 benchmark）与 HAD（前视 + 俯视双视角融合模型）合并为一篇论文：SR3M 提出问题，HAD 给出答案。主目标 RA-L + IROS 2027（约 2027-02-20 投稿），备选 Drones。

- **唯一权威的研究计划**：[docs/research_plan.md](docs/research_plan.md)（由我维护）。
- **执行合同**：[docs/protocol.md](docs/protocol.md)。当前阶段看 §17（数据集 v3 阶段 2）和 §18（STOP 探针第二轮）。
- **核心问题**：地形越复杂，俯视角对闭环地形跟随的帮助是否越大。11/04 起**论文主结果全部在数据集 v3 上**，v2 的结论降为动机和诊断。闭环地形跟随误差（§17.10 的 T1）是论文重点。
- 仓库整合长期计划：[docs/AGENT_TASK_architecture.md](docs/AGENT_TASK_architecture.md)，非当前优先。

## 3. 环境与位置

- **Python**：`/home/qlj/h3c_pro/HAD-UAV-VLN/.venv`（3.12）。shell 里的 `python` 是 miniconda，要显式用 venv。命令在仓库根目录用 `python -m experiments.<stage>` 运行，测试 `python -m pytest -q`。transformers 不装进共享 venv，需要时 `PYTHONPATH=~/sr3mxhad_runs/pylib`。
- **仿真**：主闭环为运动学闭环 K（`sim_k/`、`experiments/closed_loop_k.py`、`experiments/v3_fly.py`），用数据集的无头方式渲染（UnrealEditor-Cmd），运行前要关闭 GUI 编辑器。AirSim 路线已放弃（代码保留在工程副本 `military_mountains_airsim_p0`）。
- **阶段 2 的 UE 工程**：`E:\UnrealProjects\DesertMountain`，K profile `desert_mountain`。
  - 关卡：`/Game/Mountain/Maps/L_MTR_A–D`、`/Game/Desert/Maps/L_DES_A–D_v3`。v2 的 `L_DES_A–D` 也在其中，不改。
  - 地面是静态网格 `Terrain_*`（complex-as-simple 碰撞），没有 Landscape。
  - Blender 源文件在 `E:\blender3dProjects\{mountain,desert}`，目标表在各图的 `scene_config(_v3).json`。
- **渲染设置**：exposure_bias −10（v2 desert 实际采集值），512×512，FOV 90°。
  - 阶段 2 的控制台命令：关 VSM，加阴影偏移（11/05，protocol §17.5），代码里是 `experiments.v3_ue_check.STAGE2_CONSOLE`。
  - K 闭环要设 `SR3MXHAD_K_CONSOLE='["r.Shadow.Virtual.Enable 0","r.Shadow.CSMDepthBias 20","r.Shadow.CSMSlopeScaleDepthBias 4.5"]'`，采集和闭环必须一致，默认不开。
- **数据**：位置在 `configs/paths.toml`（不进 Git，样例 `paths.example.toml`）。
  - v2 发布划分：`<data_root>/benchmark_split_v2`（v1 只读保留）。
  - v3 试点（阶段 1）：`/mnt/f/WSL_ColdStorage/datasets/sr3m_v3_pilot`。
  - v3 阶段 2：`/mnt/f/WSL_ColdStorage/datasets/sr3m_v3`（`experiments/v3s2_capture.py` 写入，带一份 `plan.json`）。
- **实验产物（不进 Git）**：
  - v3：`runs/v3/`（试点阶段 0、导入核对 `ue_check/`）、`runs/v3/stage2/`（阶段 2 的阶段 0、路线计划、`g1/`、`shadow/`）、`runs/v3_pilot/`（阶段 1 结果、`model_check/`）。
  - v2：`runs/terrain_v2_dino/`（DINOv2 主结果）、`runs/terrain_v2/`（ResNet 与诊断）、`runs/closed_loop/`（v2 闭环）、`runs/data_audit/`（v2 数据审计）。
- **代码结构**见 README；`sr3m/` 不得 import `had`/`experiments`，`had/` 不得 import `experiments`（有测试守护）。`had/vendor/` 是 HAD@f01440a 逐字节副本，不改。

## 4. 实验纪律（违反会污染结论）

- 原始数据（`benchmark_split_v1`/`v2`、v3 试点捕获）、旧 checkpoint 只读；不得重抽划分或移动关联路线。v2 的 Blender/UE 原文件不动，v3 增补另存新文件。
- **v3 阶段 2 的数据使用**（§17）：
  - 词表、动作统计只在训练图的 train 上拟合；
  - checkpoint 和 STOP 阈值在训练图的 val_seen 上选；
  - 开发图最多查看 3 次，每次在 §17.9 登记；
  - 冻结全部配置后，测试图只打开一次。
- `PROTOCOL` 由环境变量 `SR3MXHAD_PROTOCOL` 选（v2 默认，试点 `v3_pilot`；阶段 2 的 `v3` 按 §17.5 定义，尚未写进代码），绑定 prepared 数据与特征缓存；不改已有定义，除非我同意。诊断变体（N、B0、B4）定义在 `PROTOCOL` 之外。
- run 身份绑定 git commit，**训练或渲染进行中不要提交代码**。
- 闭环：专家回放和 G1 全部通过后才能跑模型；离线预览和单元测试不等于完成真实 UE 闭环。
- 判据在运行前写进 protocol 并提交；看到结果后改判据要明确记录。
- 不另建 evidence/、verify_* 或哈希清单；Git HEAD + 工作区状态记录代码版本。

## 5. 当前状态（2026-11-05）

**阶段 2 进度**（protocol §17.11）：

- [x] §18 STOP 探针第二轮：五个变体都没过，STOP 沿用 V1（俯视 4×4、硬标签）。见 `runs/v3_pilot/model_check/results.md`。
- [x] 8 张新图建好并导入 UE，核对全部通过（`runs/v3/ue_check/check.md`）。
- [x] **阶段 0**：G0-1、G0-4、G0-6、G0-7 过；G0-2、G0-3 未过，已放行；G0-5 过。见 `runs/v3/stage2/stage0_report.md`。
  - 物体表 `configs/v3s2_objects.json`；设计 `kick_start2_relief_s2b`；计划 `runs/v3/stage2/plan.json`（1,000 条路线，62,177 个位姿）。
- [x] G1 与预览检查：8 张图都过，见 `runs/v3/stage2/g1/report.md`。
- [ ] **批量渲染**（11/05 开始，约 10–12 小时）→ 准备数据（`PROTOCOL=v3` 待写）、训练（5090）→ 开发图（≤ 3 次）→ 测试图（一次）。目标 11 月底。

**11/05 决定**（protocol §17.4、§17.5）：
- 每个目标轮流分配路线；规模不变，看数据量曲线再决定是否扩。
- 称呼一半用"外观 + 位置"，一半只用外观。
- 指令里不用东南西北：模型只知道相对起飞点的位姿。
- 同名目标不改。
- 阴影偏移 20 / 4.5。

**已确立的结论（详细见各文件）**：

- **v2 离线**：DINOv2 下 H1 14.7%（CI [9.9, 19.2]），H2 未过，对照组伤害主要来自 1 条路线；S1、S2 通过；模型几乎不用"随地形/定高"条件。见 `runs/terrain_v2_dino/decision/onepage.md`。
- **v2 闭环**：SR20 2/96；航向正反馈；根源是 v2 路线设计泄露（只看位姿的模型方向一致 93%）。见 `runs/closed_loop/k/`、`runs/data_audit/report.md`。
- **v3 试点**（阶段 1）：B 的 SR20 41.7%（门槛 50% 未过），B−N +28，航向回正。失败主要在 STOP。4×4 俯视在探针里 +21，但闭环中没有转化（B4 38.3%）。未见类别的 STOP 更差。见 `runs/v3_pilot/results.md`、`model_check/results.md`。
- **STOP**：现成图文模型（SigLIP、RemoteCLIP、Qwen3-VL 零样本）都解决不了俯视 STOP，可作为 benchmark 发现。
- **渲染**：K 的显存泄露来自 VSM，关闭后不涨且更快（`runs/infra/ue_restart/leak_report.md`）。

**论文侧未开始**：

- 核实"地形跟随误差少有人单独评测"（重点 AGC-VLN）；
- D3 真实数据离线评测；
- D4 竞品基线和 VLM 零样本基线。

## 6. 暂缓的探索路线

"轻量决策代替重型 VLM"：开放词表检测 + 深度估计 → 结构化文本状态 → 离散动作 + 置信度，低置信回退 VLM。与当前主线关系未定，**不要默认采用**。研究计划中 VLM 只作零样本基线。

## 7. 工作约定

- 改动大的东西（重构、删文件、跑长时间训练或渲染）先问我。在分支上改由我合并（当前分支 `data/v3-design`）；分支上的本地提交由 agent 判断，**没有我的命令不要推送到 GitHub**。
- 实验结论写入对应阶段的 `runs/` 目录或我指定的位置，写清配置和结论。
- 建议修改 UE 关卡或 Blender 文件之前，先在 UE 里按 actor 标签确认问题，并和 v2 关卡对照。
- 发现本文件过时，直接提出更新建议。
