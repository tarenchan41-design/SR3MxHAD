# CLAUDE.md — SR3M × HAD

> 给 agent 的项目约定。发现与代码不符时以代码为准，并提醒我更新本文件。

## 1. 我是谁、怎么合作

- PhD 中期，研究方向：无人机视觉导航 / VLA。导师放养，研究基本自主推进。
- 早上精力最好；没有明确 deadline 时容易拖。请帮我把任务拆成**有明确产出的小步骤**，每次结束时告诉我"下一步是什么"。
- 算力：本机 RTX 3070 Laptop 8GB（数据生产、闭环、少量训练）；云端 RTX 5090（训练、离线评测，不能跑仿真）。**优先低算力方案**，大规模训练前先跟我确认。

## 2. 项目全貌

SR3M（以地形为核心的无人机 VLN 仿真 benchmark）与 HAD（前视 + 俯视双视角融合模型）合并为一篇论文：SR3M 提出问题，HAD 给出答案。主目标 RA-L + IROS 2027（约 2027-02-20 投稿），备选 Drones。

- **唯一权威的研究计划**：[docs/research_plan.md](docs/research_plan.md)（由我维护）。11/10 前的唯一目标是回答"地形越复杂，俯视角帮助是否越大"；工程整合不能取代这一交付。
- **执行合同**：[docs/protocol.md](docs/protocol.md)。
- **仓库整合长期计划**：[docs/AGENT_TASK_architecture.md](docs/AGENT_TASK_architecture.md)。2026-10-21 已完成精简版结构（sr3m/had/experiments 分包、迁入 ../SR3M 依赖、路径配置）；三机对账、数据发布 v1.0、实验登记表等留到离线决策之后。

## 3. 环境与位置

- 仿真：UE 5.8.4 + Cosys-AirSim 3.5.0（插件需从源码构建），尚未完成真实闭环。
- 数据：当前发布划分 `<data_root>/benchmark_split_v2`（v1 只读保留）；地形导出见 `terrain_dir`（mountain）和 `desert_terrain_dir`（desert）；位置在 `configs/paths.toml`（不进 Git，样例 `paths.example.toml`）。v2 由 `python -m sr3m.release_v2` 生成。
- Python：`/home/qlj/h3c_pro/HAD-UAV-VLN/.venv`（3.12）。命令在仓库根目录用 `python -m experiments.<stage>` 运行，测试 `python -m pytest -q`。
- 实验产物：`runs/terrain_v2/`（不进 Git）；决策报告 `runs/terrain_v2/decision/decision.md`。`runs/terrain_v1/` 是 v1 试跑，只作参考。
- 代码结构见 README；`sr3m/` 不得 import `had`/`experiments`，`had/` 不得 import `experiments`（有测试守护）。`had/vendor/` 是 HAD@f01440a 逐字节副本，不改。

## 4. 实验纪律（违反会污染结论）

- 原始 `benchmark_split_v1`/`v2`、捕获数据、旧 checkpoint 只读；不得重抽划分或移动关联路线。v2（10/23 发布）= v1 去掉旧 desert（其余行逐字节不变）+ 新 desert 201 条：L_DES_A/B/D 进 train/val_seen，**L_DES_C 整图进 val_unseen**。新 desert 背光面很暗，已写入 v2 limitations，未剔除。
- 只用 v2 train 拟合词表、动作统计和坡度分箱。val_seen 选 checkpoint、STOP 阈值和探针正则；全部正式训练冻结后才打开 val_unseen。既有 test 不进入当前开发。
- 主实验：英文指令，A/B × seeds 42/43/44（C 暂不跑），简单拼接，不含门控、AGL 输入或图像增强。
- 不改 `experiments/protocol.py` 的 `PROTOCOL`，除非我同意：它绑定 prepared 数据与特征缓存。run 身份绑定 git commit，**训练进行中不要提交代码**。
- 闭环：16 条专家回放全部通过后才能跑模型；离线预览和单元测试不等于完成真实 UE 闭环。
- 不另建 evidence/、verify_* 或哈希清单；Git HEAD + 工作区状态记录代码版本。

## 5. 当前状态与优先事项（2026-10-26）

- 已完成（v1 数据）：prepare、十路线过拟合、train/val_seen 缓存、A/B 种子 42 试跑（`runs/terrain_v1/pilot/`，只看 val_seen）。流水线按我的要求：**每个后续实验前先讨论**。
- 判据已于开训前定稿为 H1（随地形组相对 dz 改善）+ H2（随坡度增大）+ 对照组不伤害；D 只作描述（见 protocol.md §6）。
- 10/22 决定：desert 提前重建并加入训练；决策节点移到 11/10。10/23：v2 已发布，代码已接入 v2（`PROTOCOL` = `sr3m_had_terrain_v2`，地形类别 mountain + desert 合并进 H1/H2，按类别拆分仅作描述）。v1 试跑结果不能与 v2 直接比较。
- 10/24：v2 prepare、特征缓存（`~/sr3mxhad_runs/terrain_v2/features`，2.8GB，v1 原路线特征逐位一致）、十路线过拟合（2500 步通过，MSE 0.035 / STOP F1 0.975）完成。
- 10/25：本机 3070 完成 A/B × 种子 42/43/44 正式训练（共约 8.3 小时，代码 `4c6272e` 干净工作区），6 个 run 均已在 val_seen 冻结 checkpoint 与 STOP 阈值，val_unseen 未读。输出 `runs/terrain_v2/models/`。`evaluate.py`/`summarize.py` 已改为只用 A/B（`MAIN_VARIANTS`，不改 `PROTOCOL`）。
- 10/25 21:20 打开 val_unseen（经我确认）。预注册结论：门槛通过、H1 通过（4.7%，CI [0.9, 8.1]）、H2 未通过、对照组伤害（dz +3.7cm）→ "先检查融合方式"。伤害几乎全部来自 des_c_high_dogleg_004（断崖上定高，B 跟着地形爬升）；mountain H1 9.7%、伤害 0.8cm，desert H1≈0。诊断见 `runs/terrain_v2/decision/analysis_control_harm.md`。之后的任何修改只在 val_seen 上开发与选择。
- 10/25 复查：该失败在 train/val_seen 上不复现（只在未见陡坡）；另发现 dz 泛化整体弱（val 上随地形 β 0.17–0.30，真值 0.55–0.70；val_seen dz 在第 2–5 epoch 最好后过拟合）。我选择**方向 A：先解决 dz 泛化**，偏离预注册规则的"先查融合"，理由与计划见 `runs/terrain_v2/decision/discussion_step2.md`。A2 已完成（`runs/terrain_v2/decision/a2_results.md`）：几何 oracle 可消掉 54% 的 dz 误差，冻结全局池化特征线性只读出 3–11%，策略 13–18%，**瓶颈主要在表示**。A3 探针已完成（`runs/terrain_v2/decision/a3_results.md`）：留一图开发集上，外观特征（G/R1）对未见地图 dz 几乎无效（−7%…+6%），Depth Anything V2 深度特征 R2 可迁移（≈10%，与 G 拼接 12%），离几何 oracle 53% 仍远；val_seen 排序与未见地图相反，不能用来选表示。10/26 我同意进入 v3 单折最小对照：`PROTOCOL` = `sr3m_had_terrain_v3`（新增 AD/BD 深度变体与深度规格，protocol.md §9），留出 L_MTN_B + L_DES_B，B vs BD 各 seed 42。v2 的 prepared/缓存/结果保留，但当前代码不再读 v2 准备集（复现 v2 用提交 `d7ba299`）。
- 10/26 v3 单折（代码 `a62ad92`）：留出 L_MTN_B + L_DES_B，BD 相对 B 的随地形 dz 只改善 2–3%（CI 跨 0），β 0.13→0.20（真值约 0.6），**不跑完整 v3**，见 `runs/terrain_v2/decision/v3_fold_B.md`。我同意先做最后一个低成本检查：俯视**度量**深度探针（留一图，只探针不训练）；增益 < 20% 就收束方向 A，把"未见地形上的高度控制泛化难"写成 benchmark 发现。研究计划已写入 D1–D4（投稿目标、定位、真实数据离线评测、竞品基线）。

## 6. 暂缓的探索路线

"轻量决策代替重型 VLM"：开放词表检测 + 深度估计 → 结构化文本状态（Task/Ego/Clearance/Objects/Memory）→ 离散动作 + 置信度，低置信回退 VLM；关键对照为仿真真值状态 vs 感知状态。与当前主线关系未定，**不要默认采用**。研究计划中 VLM 只作 API 零样本基线。

## 7. 工作约定

- 改动大的东西（重构、删文件、跑长时间训练）先问我。在分支上改由我合并；分支上的本地提交由 agent 判断，**没有我的命令不要推送到 GitHub**。
- 实验结论写入 `runs/terrain_v2/decision/` 或我指定的位置，写清配置和结论。
- 发现本文件过时，直接提出更新建议。
