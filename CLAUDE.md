# CLAUDE.md — SR3M × HAD

> 给 agent 的项目约定。发现与代码不符时以代码为准，并提醒我更新本文件。

## 1. 我是谁、怎么合作

- PhD 中期，研究方向：无人机视觉导航 / VLA。导师放养，研究基本自主推进。
- 早上精力最好；没有明确 deadline 时容易拖。请帮我把任务拆成**有明确产出的小步骤**，每次结束时告诉我"下一步是什么"。
- 算力：
  - 本机 RTX 3070 Laptop 8GB：数据生产、UE 渲染、闭环、训练。阶段 2 的训练瓶颈在 CPU/memmap，3070 每个 epoch 约 45 s，不比 5090 慢（11/07 起 D4 和数据量曲线都在本机训练）。WSL 只有 5.9 GB 内存：抽特征 `--workers 2`，同一时间只跑一个训练。
  - 云端 RTX 5090：训练、抽特征、离线评测，不能跑仿真。由我手动 clone 并运行，agent 每次给出具体命令。训练不能跨 GPU 逐位复现，但差异在种子波动之内（§17.9 硬件对照）。
  - **优先低算力方案**，大规模训练前先跟我确认。
- 流水线按我的要求：**每个后续实验前先讨论**。
- 实验中发现的现象可以发展成新的论文点（不必都事先预注册），但要标明"事后发现、仅描述"，和预注册判据分开写。

## 2. 项目全貌

SR3M（以地形为核心的无人机 VLN 仿真 benchmark）与 HAD（前视 + 俯视双视角融合模型）合并为一篇论文：SR3M 提出问题，HAD 给出答案。主目标 RA-L + IROS 2027（约 2027-02-20 投稿），备选 Drones。

- **唯一权威的研究计划**：[docs/research_plan.md](docs/research_plan.md)（由我维护）。
- **执行合同**：[docs/protocol.md](docs/protocol.md)。阶段 2 的判据与结果看 §17（§17.9 开发图、§17.10 测试图）。
- **核心问题**：地形越复杂，俯视角对闭环地形跟随的帮助是否越大。11/04 起**论文主结果全部在数据集 v3 上**，v2 的结论降为动机和诊断。闭环地形跟随误差（§17.10 的 T1）是论文重点。
  - 11/07 测试图结果：俯视帮助闭环地形跟随和到达（T1–T3 过），但"越复杂帮助越大"（H2）未获支持；帮助来自俯视本身，双视角融合、HAD 融合结构都没有显示额外贡献（事后发现）。论文主张如何限定待讨论。
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
  - 采集和闭环必须一致。阶段 2 的飞行脚本（`experiments/v3s2_fly.py`、`v3s2_test.py`）直接传 `STAGE2_CONSOLE`，并核对路线采集时的设置；其他 K 入口仍需设 `SR3MXHAD_K_CONSOLE='["r.Shadow.Virtual.Enable 0","r.Shadow.CSMDepthBias 20","r.Shadow.CSMSlopeScaleDepthBias 4.5"]'`，默认不开。
- **数据**：位置在 `configs/paths.toml`（不进 Git，样例 `paths.example.toml`）。
  - v2 发布划分：`<data_root>/benchmark_split_v2`（v1 只读保留）。
  - v3 试点（阶段 1）：`/mnt/f/WSL_ColdStorage/datasets/sr3m_v3_pilot`。
  - v3 阶段 2：`/mnt/f/WSL_ColdStorage/datasets/sr3m_v3`（`experiments/v3s2_capture.py` 写入，带一份 `plan.json`）。
- **实验产物（不进 Git）**：
  - v3：`runs/v3/`（试点阶段 0、导入核对 `ue_check/`）、`runs/v3_pilot/`（阶段 1 结果、`model_check/`）。
  - v3 阶段 2：`runs/v3/stage2/`，包括阶段 0、路线计划、`g1/`、`shadow/`、`prepared/`（准备数据），以及：
    - `models/`（主实验与 D4 的 checkpoint）、`models_frac25/`、`models_frac50/`、`models_3070check/`；
    - `dev_look1/`、`dev_look2/`（开发图两次查看）；
    - `test/`（测试图：`report.md`、`summary.json`、`flights/`、`offline/`）。
  - v3 特征缓存：`~/sr3mxhad_runs/v3/`，包括 `features`（2×2）、`features_grid4`（俯视 4×4）、`features_mosaic`（AV 拼图）和 `ground`（地面高度）。
  - v2：`runs/terrain_v2_dino/`（DINOv2 主结果）、`runs/terrain_v2/`（ResNet 与诊断）、`runs/closed_loop/`（v2 闭环）、`runs/data_audit/`（v2 数据审计）。
- **代码结构**见 README；`sr3m/` 不得 import `had`/`experiments`，`had/` 不得 import `experiments`（有测试守护）。`had/vendor/` 是 HAD@f01440a 逐字节副本，不改。

## 4. 实验纪律（违反会污染结论）

- 原始数据（`benchmark_split_v1`/`v2`、v3 试点捕获）、旧 checkpoint 只读；不得重抽划分或移动关联路线。v2 的 Blender/UE 原文件不动，v3 增补另存新文件。
- **v3 阶段 2 的数据使用**（§17）：
  - 词表、动作统计只在训练图的 train 上拟合；
  - checkpoint 和 STOP 阈值在训练图的 val_seen 上选；
  - 开发图最多查看 3 次，每次在 §17.9 登记（已用 2 次）；
  - 冻结全部配置后，测试图只打开一次（**11/07 已打开**，不再重飞或改判据；新实验要在测试图上跑，必须先和我讨论并明确标为事后）。
- `PROTOCOL` 由环境变量 `SR3MXHAD_PROTOCOL` 选（v2 默认，试点 `v3_pilot`，阶段 2 用 `v3`），绑定 prepared 数据与特征缓存；不改已有定义，除非我同意。诊断变体和基线变体定义在 `PROTOCOL` 之外：N、B0、B4、C4、B40（B4 去掉位姿），以及 D4 的 AV、DG。阶段 2 选了俯视 4×4，论文里的 B/C/B0 在代码里是 B4/C4/B40。
- run 身份绑定 git commit，**训练或渲染进行中不要提交代码**。
- 闭环：专家回放和 G1 全部通过后才能跑模型；离线预览和单元测试不等于完成真实 UE 闭环。
- 判据在运行前写进 protocol 并提交；看到结果后改判据要明确记录。
- 不另建 evidence/、verify_* 或哈希清单；Git HEAD + 工作区状态记录代码版本。

## 5. 当前状态（2026-11-08）

**阶段 2 进度**（protocol §17.11）：

- [x] §18 STOP 探针第二轮：五个变体都没过，STOP 沿用 V1（俯视 4×4、硬标签）。见 `runs/v3_pilot/model_check/results.md`。
- [x] 8 张新图建好并导入 UE，阶段 0、G1 与预览检查都已完成。见 `runs/v3/stage2/stage0_report.md`、`g1/report.md`。
- [x] 批量渲染（1,000 条路线）→ 准备数据（`PROTOCOL=v3`）→ 抽特征 → 训练：B4/A/C4/N/AV/DG 各 3 个种子，加 B40、数据量曲线 25%/50%。
- [x] 开发图第 1 次查看：选俯视 4×4（B4 95.0% 对 B 91.2%），开发门槛过。第 2 次查看：D4 流水线检查，正常。
- [x] **测试图（11/07，`0395e44`）**：T1、T2、T3 全过，见 `runs/v3/stage2/test/report.md`、protocol §17.10。
- [ ] 讨论论文主张怎么限定；决定剩余实验（D3 真实数据、可选阶段 3 平坦场景）；写作准备。

**11/05 决定**（protocol §17.4、§17.5）：
- 每个目标轮流分配路线；规模不变，看数据量曲线再决定是否扩。
- 称呼一半用"外观 + 位置"，一半只用外观。
- 指令里不用东南西北：模型只知道相对起飞点的位姿。
- 同名目标不改。
- 阴影偏移 20 / 4.5。

**已确立的结论（详细见各文件）**：

- **v3 测试图**（论文主结果，3 个种子，K 闭环，120 条路线）：
  - **T1** 地形跟随 R_cl 0.275 [0.153, 0.398]，三个种子都 > 0；
  - **T2** SR20 B 90.8% 对 A 49.7%，差 +41.1 个百分点 [+33.1, +48.9]；
  - **T3** 定高路线不伤害；
  - S2（C 对 A）、离线 H1（10.7%）通过；
  - H2 未获支持：起伏三档的 R_cl 为 0.54 / 0.17 / 0.24；
  - 数据量曲线 SR20 73% / 84% / 93%，还没饱和；
  - D4：B − AV +19.2 个百分点，B − DG 不显著。
  - **事后发现**：
    - B ≈ C（只看俯视），B ≈ DG（token 融合）；
    - B 的 E 6.4 m 对专家 1.6 m，β 约 0.5 对 0.98，闭环地形跟随远未解决。

- **v2 离线**：DINOv2 下 H1 14.7%（CI [9.9, 19.2]），H2 未过，对照组伤害主要来自 1 条路线；S1、S2 通过；模型几乎不用"随地形/定高"条件。见 `runs/terrain_v2_dino/decision/onepage.md`。
- **v2 闭环**：SR20 2/96；航向正反馈；根源是 v2 路线设计泄露（只看位姿的模型方向一致 93%）。见 `runs/closed_loop/k/`、`runs/data_audit/report.md`。
- **v3 试点**（阶段 1）：B 的 SR20 41.7%（门槛 50% 未过），B−N +28，航向回正。失败主要在 STOP。4×4 俯视在探针里 +21，但闭环中没有转化（B4 38.3%）。未见类别的 STOP 更差。见 `runs/v3_pilot/results.md`、`model_check/results.md`。
- **STOP**：现成图文模型（SigLIP、RemoteCLIP、Qwen3-VL 零样本）都解决不了俯视 STOP，可作为 benchmark 发现。
- **渲染**：K 的显存泄露来自 VSM，关闭后不涨且更快（`runs/infra/ue_restart/leak_report.md`）。

**论文侧**：

- [x] D4 竞品基线：AV、DG 已完成。VLM 零样本基线未做。
- [ ] 核实"地形跟随误差少有人单独评测"（重点 AGC-VLN；11/05 的 AGL 指标重叠检查未发现重叠）；
- [ ] D3 真实数据离线评测。

## 6. 暂缓的探索路线

"轻量决策代替重型 VLM"：开放词表检测 + 深度估计 → 结构化文本状态 → 离散动作 + 置信度，低置信回退 VLM。与当前主线关系未定，**不要默认采用**。研究计划中 VLM 只作零样本基线。

## 7. 工作约定

- 改动大的东西（重构、删文件、跑长时间训练或渲染）先问我。在分支上改由我合并（当前分支 `data/v3-design`）；分支上的本地提交由 agent 判断，**没有我的命令不要推送到 GitHub**。
- 实验结论写入对应阶段的 `runs/` 目录或我指定的位置，写清配置和结论。
- 建议修改 UE 关卡或 Blender 文件之前，先在 UE 里按 actor 标签确认问题，并和 v2 关卡对照。
- 发现本文件过时，直接提出更新建议。
