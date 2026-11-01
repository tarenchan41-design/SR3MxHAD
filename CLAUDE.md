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

- 仿真：主闭环为运动学闭环 K（`sim_k/`、`experiments/closed_loop_k.py`、`experiments/k_gates.py`），用数据集的无头方式渲染，运行时要关闭 GUI 编辑器。UE 5.8.4 + Cosys-AirSim 3.5.0 已构建在工程副本 `military_mountains_airsim_p0`，代码保留但不作主结果。
- 数据：当前发布划分 `<data_root>/benchmark_split_v2`（v1 只读保留）；地形导出见 `terrain_dir`（mountain）和 `desert_terrain_dir`（desert）；位置在 `configs/paths.toml`（不进 Git，样例 `paths.example.toml`）。v2 由 `python -m sr3m.release_v2` 生成。
- Python：`/home/qlj/h3c_pro/HAD-UAV-VLN/.venv`（3.12）。命令在仓库根目录用 `python -m experiments.<stage>` 运行，测试 `python -m pytest -q`。
- 实验产物（不进 Git）：当前 DINOv2 重做在 `runs/terrain_v2_dino/`（缓存 `~/sr3mxhad_runs/terrain_v2_dino/features`）；v2 ResNet 结果与全部诊断在 `runs/terrain_v2/`，决策报告 `runs/terrain_v2/decision/decision.md`。`runs/terrain_v1/` 是 v1 试跑，只作参考。cache/overfit/closed_loop 需要 `PYTHONPATH=~/sr3mxhad_runs/pylib`（transformers 不装进共享 venv）。
- 代码结构见 README；`sr3m/` 不得 import `had`/`experiments`，`had/` 不得 import `experiments`（有测试守护）。`had/vendor/` 是 HAD@f01440a 逐字节副本，不改。

## 4. 实验纪律（违反会污染结论）

- 原始 `benchmark_split_v1`/`v2`、捕获数据、旧 checkpoint 只读；不得重抽划分或移动关联路线。v2（10/23 发布）= v1 去掉旧 desert（其余行逐字节不变）+ 新 desert 201 条：L_DES_A/B/D 进 train/val_seen，**L_DES_C 整图进 val_unseen**。新 desert 背光面很暗，已写入 v2 limitations，未剔除。
- 只用 v2 train 拟合词表、动作统计和坡度分箱。val_seen 选 checkpoint、STOP 阈值和探针正则；全部正式训练冻结后才打开 val_unseen。既有 test 不进入当前开发。
- 主实验：英文指令，A/B × seeds 42/43/44，简单拼接，不含门控、AGL 输入或图像增强。10/27 起视觉骨干为冻结 DINOv2-B（protocol.md §11，判据已预注册）；C、N 只作诊断。
- 不改 `experiments/protocol.py` 的 `PROTOCOL`，除非我同意：它绑定 prepared 数据与特征缓存。run 身份绑定 git commit，**训练进行中不要提交代码**。
- 闭环：16 条专家回放全部通过后才能跑模型；离线预览和单元测试不等于完成真实 UE 闭环。
- 不另建 evidence/、verify_* 或哈希清单；Git HEAD + 工作区状态记录代码版本。

## 5. 当前状态与优先事项（2026-10-27）

- 已完成（v1 数据）：prepare、十路线过拟合、train/val_seen 缓存、A/B 种子 42 试跑（`runs/terrain_v1/pilot/`，只看 val_seen）。流水线按我的要求：**每个后续实验前先讨论**。
- 判据已于开训前定稿为 H1（随地形组相对 dz 改善）+ H2（随坡度增大）+ 对照组不伤害；D 只作描述（见 protocol.md §6）。
- 10/22 决定：desert 提前重建并加入训练；决策节点移到 11/10。10/23：v2 已发布，代码已接入 v2（`PROTOCOL` = `sr3m_had_terrain_v2`，地形类别 mountain + desert 合并进 H1/H2，按类别拆分仅作描述）。v1 试跑结果不能与 v2 直接比较。
- 10/24：v2 prepare、特征缓存（`~/sr3mxhad_runs/terrain_v2/features`，2.8GB，v1 原路线特征逐位一致）、十路线过拟合（2500 步通过，MSE 0.035 / STOP F1 0.975）完成。
- 10/25：本机 3070 完成 A/B × 种子 42/43/44 正式训练（共约 8.3 小时，代码 `4c6272e` 干净工作区），6 个 run 均已在 val_seen 冻结 checkpoint 与 STOP 阈值，val_unseen 未读。输出 `runs/terrain_v2/models/`。`evaluate.py`/`summarize.py` 已改为只用 A/B（`MAIN_VARIANTS`，不改 `PROTOCOL`）。
- 10/25 21:20 打开 val_unseen（经我确认）。预注册结论：门槛通过、H1 通过（4.7%，CI [0.9, 8.1]）、H2 未通过、对照组伤害（dz +3.7cm）→ "先检查融合方式"。伤害几乎全部来自 des_c_high_dogleg_004（断崖上定高，B 跟着地形爬升）；mountain H1 9.7%、伤害 0.8cm，desert H1≈0。诊断见 `runs/terrain_v2/decision/analysis_control_harm.md`。之后的任何修改只在 val_seen 上开发与选择。
- 10/25 复查：该失败在 train/val_seen 上不复现（只在未见陡坡）；另发现 dz 泛化整体弱（val 上随地形 β 0.17–0.30，真值 0.55–0.70；val_seen dz 在第 2–5 epoch 最好后过拟合）。我选择**方向 A：先解决 dz 泛化**，偏离预注册规则的"先查融合"，理由与计划见 `runs/terrain_v2/decision/discussion_step2.md`。A2 已完成（`runs/terrain_v2/decision/a2_results.md`）：几何 oracle 可消掉 54% 的 dz 误差，冻结全局池化特征线性只读出 3–11%，策略 13–18%，**瓶颈主要在表示**。A3 探针已完成（`runs/terrain_v2/decision/a3_results.md`）：留一图开发集上，外观特征（G/R1）对未见地图 dz 几乎无效（−7%…+6%），Depth Anything V2 深度特征 R2 可迁移（≈10%，与 G 拼接 12%），离几何 oracle 53% 仍远；val_seen 排序与未见地图相反，不能用来选表示。10/26 我同意进入 v3 单折最小对照：`PROTOCOL` = `sr3m_had_terrain_v3`（新增 AD/BD 深度变体与深度规格，protocol.md §9），留出 L_MTN_B + L_DES_B，B vs BD 各 seed 42。v2 的 prepared/缓存/结果保留。
- 10/26 v3 单折（代码 `a62ad92`）：留出 L_MTN_B + L_DES_B，BD 相对 B 的随地形 dz 只改善 2–3%（CI 跨 0），β 0.13→0.20（真值约 0.6），**不跑完整 v3**，见 `runs/terrain_v2/decision/v3_fold_B.md`。我同意先做最后一个低成本检查：俯视**度量**深度探针（留一图，只探针不训练）；增益 < 20% 就收束方向 A，把"未见地形上的高度控制泛化难"写成 benchmark 发现。研究计划已写入 D1–D4（投稿目标、定位、真实数据离线评测、竞品基线）。
- 10/26 **方向 A 已收束**（我确认）：现成单目度量深度（Depth Anything V2 Metric-Outdoor）在俯视/高空下给不出 AGL（估计恒为 2–10m，与真值不相关，尺度来自驾驶场景先验），见 `runs/terrain_v2/decision/m1_results.md`。结论：所测的冻结视觉特征（ResNet 外观、相对深度）在未见地图上只能拿到约 10% 的 dz 改善（几何 oracle 45–55%）。**更正**：oracle 拆分（`m1_oracle_split.py`）表明 AGL 对 dz 没有贡献，全部信息来自沿航向前方的地形起伏（带符号）；水平步长恒为约 2m，所以需要的是与尺度无关的沿航向坡度，不是绝对尺度，"单目做不到"不能写进论文。作为 benchmark 发现：所测冻结特征读不出沿航向坡度；更擅长 3D 结构的编码器（DINOv2/表面法向/VLM 视觉塔）未测试。剩余时间集中在核心问题"地形越复杂，俯视角帮助是否越大"。代码已按我的决定恢复到 v2（与 `d7ba299` 代码逐字节相同，`PROTOCOL` = `sr3m_had_terrain_v2`；v3 实现保留在 `a62ad92`）。我同意补跑 C（仅俯视）× 种子 42/43/44 作诊断，输出 `runs/terrain_v2/models_c/`。
- 10/26 C 诊断完成（代码 `4c8dcc9`，见 `runs/terrain_v2/decision/c_results.md`，仅描述）：单独俯视不比单独前视好（val_unseen 随地形 A→C 0.0%），B 同时优于 A 和 C（C→B 4.7%，CI [1.6, 7.5]）——帮助来自双视角互补；desert 定高伤害与俯视输入有关（C +3.3cm、B +6.6cm）；平移/yaw/STOP 三个变体几乎相同，水平导航主要由指令和状态决定。
- 10/26 一页结论已写（`runs/terrain_v2/decision/onepage.md`，待我审阅）。我同意加无视觉基线 N 与编码器探针。N 定义在 `PROTOCOL` 之外（`DIAGNOSTIC_VARIANTS`，改 `PROTOCOL` 会使 v2 准备集/缓存/checkpoint 全部失效，所以 `PROTOCOL` 未改），见 protocol.md §10；先跑种子 42，输出 `runs/terrain_v2/models_n/`。两者都等 3070 空出来再跑（编码器探针也可预缩图后放到云端 5090）。
- 10/26 N（无视觉，种子 42，代码 `0b1b0c5`）完成，见 `runs/terrain_v2/decision/n_results.md`：转弯/STOP/非地形路线水平误差与 A/B/C 持平，**离线水平导航不需要视觉**（teacher forcing 下指令 + 状态足够），视觉作用集中在 dz；随地形 dz 相对固定直飞 N 7.7%、A 12.5%、B 16.7%。论文需把 N 作为必报基线、水平导航靠闭环检验。
- 10/26 E1 编码器探针完成（留一图、只用视觉特征，见 `runs/terrain_v2/decision/e1_results.md`）：ResNet 0.9%，SigLIP-so400m（VLM 视觉塔）7.4%，DINOv2-B 16.9%，DA-V2-S 编码器 16.4%，**DINOv2-B + 相对深度 20.4%**（勉强过事先定的 20%，22 组取最好、偏乐观）；DINOv2 仅俯视 13.9%，俯视价值被 ResNet 掩盖。下一步待我决定：是否用 DINOv2-B 重建缓存并重训 A/B/N（需改 `PROTOCOL`，约 12 小时）。一页结论已补 N 与 E1。
- 10/27 我同意用 DINOv2 重做（protocol.md §11 预注册：主判据同 §6；次判据 S1 编码器是否放大俯视帮助、S2 俯视单独是否优于前视）。`PROTOCOL` = `sr3m_had_terrain_v2_dinov2`；只换冻结视觉特征（CLS + 2×2 patch 平均，3840 维，float16 缓存），其余同 v2。v2 ResNet 代码在 `4b0faf9`。代码 `f295e31` 跑完：prepare、过拟合、缓存、A/B × 3 种子（10/27 09:58 冻结）。
- 10/27 10:09 打开 val_unseen（第二次查看，经我确认），见 `runs/terrain_v2_dino/decision/dino_results.md`：门槛过，**H1 14.7%（CI [9.9, 19.2]）**，H2 未过（8.7%，CI [−3.0, 23.1]），对照组伤害 +2.4cm（仍主要是 des_c_high_dogleg_004，40.7cm）→ §6 决策表同 v2"先查融合"。**S1 通过**（Δ = R_dino − R_resnet = +10.0%，CI [5.3, 14.7]）→ 论文主结果用 DINOv2 版，ResNet 作对照。**S2 通过**（C 相对 A 11.5%，CI [5.7, 16.8]），"俯视单独优于前视"可写；S3：相对 N，A 6.5%、C 17.2%、B 20.3%。
- 10/27 融合排查（`runs/terrain_v2_dino/decision/fusion_diag.md`，仅描述）：A/B 几乎不用指令里的"随地形/定高"条件（反事实互换只让 β 变化约 0.07）；未见陡坡定高的误爬升来自俯视。我同意 R1（protocol.md §12，`train.py --augment hold`，代码 `f9a20ce`）：随地形路线加定高指令副本、dz 目标 0；开发判据只用 val_seen（反事实定高 β ≤ 0.05 等三条）。B、A 种子 42，输出 `runs/terrain_v2_dino/r1/`。
- 10/27 R1 结果（`runs/terrain_v2_dino/r1/r1_results.md`）：服从性修好（反事实定高 β 0.24→0.00），但未过开发判据（B 随地形 β 0.337→0.301；A 随地形 dz +1.0cm），不跑 3 种子。**我决定融合问题写成发现、不再修**（"模型不用高度条件"+反事实互换测试+R1 的权衡），转向闭环仿真。一页结论 DINOv2 版：`runs/terrain_v2_dino/decision/onepage.md`（10/28 我已审阅）。
- 10/27 偏离诊断会议（`~/.claude/drift-reports/.../report.md`）后：research_plan 已按现状重写（`bec03cd`，含"已确立的结论"和当前阶段）；闭环先用 16 条 val_seen 路线跑通，未见地图闭环测试集等 P2 后再定；恢复逐路线分析；N 补种子 43/44（10/28 完成，S3 三种子：A 6.7%、C 17.4%、B 20.4%）。
- 10/28 闭环路线：**先搭 A（Cosys-AirSim，P0 由我在 Windows 上构建，时间盒 1 天），同时备好 K（运动学闭环，复用采集渲染器）**；A 构建失败或 P1 图像对不上就转 K。理由：想用动力学作差异点，K 与 A 共用 WSL 驱动，可量化动力学 gap。材料在 `runs/closed_loop/`（`p0_checklist.md`、`check_airsim.py`、`k_spec.md`）。注意：SR3M 数据由工作台 SceneCapture2D 管线采集（手动曝光，desert +4 EV、其余 −10 EV），不是 AirSim 相机。
- 10/29–30 AirSim P0 通过（修了 `moveToPosition` 滑过目标约 1 m，改 velocity servo，`c3ec51e`；单步 0.14 m、0.005°）。yaw 判据在看到结果后放宽：3 s 后的漂移不再判，只判观测时刻，论文要写明。
- 10/30 P1 查明（`runs/closed_loop/p1_check/p1_status.md`）：AirSim 发白是插件 sRGB 渲染目标多了两层 gamma，已能修；用数据集代码无头重采可逐像素复现数据集（同顺序 MAE 0.00）。GUI、PIE 与数据集的差距来自运行方式：关天光后接近（只有 2 个位姿的证据，机制未证实），地面纹理差异原因未查。**余弦不能单独当判据**（AirSim 俯视中位 0.70，B 的 dz 只变 2.71 cm）。事先定的 P1 判据：B 的 |Δdz| 中位数 < 2.4 cm 且 B−A 变化 < 2.4 cm；AirSim 关天光 2.71 cm 未过（差距小），无头路径 0.08 和 1.16 cm 过。**我决定转 K**，放弃动力学差异点，论文写"运动学闭环"（protocol.md §7）。
- 10/30 K 门槛全过（分支 `closedloop/kinematic`，结果在 `runs/closed_loop/k/`）：G1 B 的 |Δdz| 中位 0.12 cm、p90 0.82 cm，B−A 变化 0.02 cm（16 路线 512 位姿，乱序）；G2 最大 0.02 cm；G3 16/16、零碰撞。实测约 1.9 s/步（含每 32 步重启）。冒烟：B 种子 42 飞 `mtn_b_low_arc_005` 预算耗尽未 STOP，离终点 56 m、进度 39%，只有 1 条路线，不下结论。还未回答：数据集采集时天光基本不起作用这一点，对已有结果（尤其 desert 暗面、H2 暗帧混淆）的影响。
- 10/30 K 首轮飞行（代码 `e497d67`，A/B × 3 种子 × 16 条 val_seen，5.5 小时，见 `runs/closed_loop/k/flights/results.md`）：**96 次 SR20 只成功 2 次**，终点距离中位 A 142 m、B 128 m。K 链路无误（第 0 步与离线一致）。不是直飞塌缩，但转向不够准，71/96 比直飞基线更远；中位在专家步数 46% 处已偏离 20 m 以上。只把离线误差积分也只有约 25–29 m，走偏后恶化。B/A 差异样本太稀，不作结论。
- 10/30 N 闭环（代码 `90c74b4`，3 种子，同文件，仅描述）：SR20 0/48，也飞不到；进度 0.20（A 0.25、B 0.29），STOP 25/48（A 36、B 40），碰撞或撞地 6（A 3、B 0）。水平控制的主要问题与视觉无关，但闭环里视觉并非完全没用。SR20 分不出差异。
- 10/30 我决定先试短程滚动（规格 `runs/closed_loop/k/rollout_spec.md`，代码 `experiments/short_rollout.py`，待我确认后跑）。**但完整论文最终很可能还是要面对长程 SR20**：该领域的前沿论文都报告闭环 SR20，短程指标只解决"测不到高度问题"，不能替代。
- 10/31 短程滚动结果（代码 `efcd53c`，`runs/closed_loop/k/rollouts/results.md`，val_seen 仅描述）：事先定的判据未过（N−B CI 跨 0）。随地形 20 m 窗口高度误差 N 3.05、A 3.19、B 3.10 m，不爬升基线 3.27 m，爬升 β 0.07–0.10。离线累加也有 2.8–3.0 m，所以弱的主要不是闭环；这批路线离线也分不出 N、A、B。定高路线 N 比 B 差约 7 cm（CI 不跨 0）。下一步待我决定。
- 10/31 训练路线飞行（B 种子 42 × 16 条匹配路线，代码 `93e4328`，`runs/closed_loop/k/flights_train/results.md`，仅描述）：SR20 同样 0/16，进度 0.25（val_seen 0.33，配对差 CI 跨 0）。离线转向在 train 上明显更好（yaw MAE 0.32° vs 0.67°），离线误差累加 11/16 在 20 m 内，但第 50 步离线累加只差 2.8 m、3°，实际飞行已差 26 m、46°。**SR20 低的主因不是泛化，差距出现在闭环**；"专家数据没有回到路径的样本"是推断，未验证。下一步待我决定。
- 11/01 回正探针（代码 `feeb63b`，规格与结果在 `runs/closed_loop/k/recovery_spec.md`、`recovery/results.md`，训练路线，B、N 种子 42，仅描述）：事先判据未过，**B 不能可靠回正**。航向是正反馈：偏右 10° 继续多右转 0.62°/步（判据 ≥0.5 回正，只有 21% 点对转回），16/16 路线增益为负，路线前段最重，谱半径 1.03。转向响应来自状态、不来自图像（只换图 0.00°/步；只换状态 −0.71；N −0.60）。横向方向对但只有判据的一半。下一步待我决定。
- 11/01 我决定排查数据质量（分支 `data/audit-v2`，只读，时间盒 1 天，`runs/data_audit/report.md`，范围 train + val_seen），R2 状态扰动暂缓。标注和指令的数值属实（终点方位 606/606，距离误差中位 0.1 m）。设计问题：几乎全部指令靠东南西北或图像方位，模型没有绝对朝向；40% 路线形状不由指令决定；终点时地标在画面内只有 54%（上限）；俯视 22% 的帧近乎单色；industry 俯视暗帧比 desert 还多，limitations 未写。这些目前解释不了训练路线也飞不到。`human_review` 全部 pending，抽检页 `runs/data_audit/review/index.html` 待我打分。
- 10/28 逐路线分析（`runs/terrain_v2_dino/decision/per_route.md`，论文图 `fig_paper/`，val_unseen 已冻结报告，仅描述）：俯视相对增益不随坡度变（中位 13–17%），绝对增益随坡度变大（2.4→4.7→7.6 cm）；大幅负增益只在平缓路线；低空收益最大（mountain 低空 27%），高空小（desert 高空 4 条 −1%）——事后假设，需在新测试集上预注册才能写成结论。

## 6. 暂缓的探索路线

"轻量决策代替重型 VLM"：开放词表检测 + 深度估计 → 结构化文本状态（Task/Ego/Clearance/Objects/Memory）→ 离散动作 + 置信度，低置信回退 VLM；关键对照为仿真真值状态 vs 感知状态。与当前主线关系未定，**不要默认采用**。研究计划中 VLM 只作 API 零样本基线。

## 7. 工作约定

- 改动大的东西（重构、删文件、跑长时间训练）先问我。在分支上改由我合并；分支上的本地提交由 agent 判断，**没有我的命令不要推送到 GitHub**。
- 实验结论写入 `runs/terrain_v2/decision/` 或我指定的位置，写清配置和结论。
- 发现本文件过时，直接提出更新建议。
