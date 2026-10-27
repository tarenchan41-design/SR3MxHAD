# SR3M × HAD 实验协议

本文件是实验的执行合同：数据、动作坐标、STOP、缓存、评测、离线决策和闭环流程。研究动机与时间线见 [research_plan.md](research_plan.md)。代码中的常量以 `experiments/protocol.py` 的 `PROTOCOL` 为准；改动 `PROTOCOL` 会使已有 prepared 数据和特征缓存失效。

## 1. 实验数据与语言

> 10/23：`benchmark_split_v2` 已发布（`python -m sr3m.release_v2`），正式训练与 11/10 决策都在 v2 上进行。v2 = v1 去掉旧 desert 的全部 201 条（train/val_seen/test = 129/23/49，捕获保留不删），其余 v1 行逐字节不变；加入重建的 desert 4 图 201 条：L_DES_A/B/D 按 v1 同一规则（同种子、15% 路线组配额）分 train 130 / val_seen 23，**L_DES_C 整图 48 条进 val_unseen**。分配代码复现 v1 全部 1606 条（0 差异）。v1 试跑（`runs/terrain_v1/pilot/`）只作参考，不与 v2 结果比较。

v2 划分为 train/val_seen/val_unseen/test = 1028/184/247/147 条，主实验直接使用，不再过滤。prepare 后 v1 原有的 898/161/199 条路线在 v2 中逐条一致（标签与数组均相同）。原始数据和地形导出的位置见 `configs/paths.toml`，均只读。

新 desert 地形起伏大、太阳高度角低（俯仰 −32°…−40°），背光面能见度很低：201 条中 77 条有黑色像素 >20% 的帧（共 6174 帧，L_DES_B 最多；mountain 为 17 条），9 条自动质检 failed（全部 near_black）。确认是场景光照所致（连续离地证书通过、采集无中断或降速、UE 视口同样昏暗），按原样发布，写入 v2 `scene_registry.json` limitations。

英文译文在捕获文件和索引的 `instruction`，中文原文在 `instruction_zh`。读取器验证索引、捕获与 source-plan 的一致性；分组使用中文原文，策略使用英文。仅用 1028 条训练指令拟合词表，实测 1,439 项、上限 256 tokens，三集合均无截断。val_seen/unseen 未知 token 占比约 0.507%/2.763%，不反向扩充词表。

| 集合 | 观测 | 原始转移 | STOP 正样本（含末观测） | 有效动作 |
|---|---:|---:|---:|---:|
| train | 127856 | 126828 | 6171 | 121685 |
| val_seen | 22336 | 22152 | 1104 | 21232 |
| val_unseen | 29984 | 29737 | 1484 | 28500 |

prepare 不读取既有 test 的捕获与动作，只在发布索引身份检查中包含其 JSONL。test 不参与本轮选模与开发；新 desert 的留出地图进入 val_unseen 主评测。

## 2. 动作坐标与 yaw

动作合同 `capture_adjacent_yaw_fru_m_rad_v1` 沿用真实相邻记录。源 UE 为左手世界系、Z 向上；引擎长度是厘米，捕获的 `position_m` 已是米，不能再除以 100。源 yaw 为度，正向从 +X 转向 +Y。

令 `psi_t = radians(rotation_deg.yaw)`，`Delta = p[t+1]-p[t]`：

```text
dx =  cos(psi_t)*Delta.x + sin(psi_t)*Delta.y
dy = -sin(psi_t)*Delta.x + cos(psi_t)*Delta.y
dz = Delta.z
dyaw = wrap(psi[t+1]-psi[t])             # [-pi, pi)
```

动作是当前 yaw 的前/右/上，米/弧度；dyaw 正值向右转。图上显示度，Cosys-AirSim 角度接口入参为度。旧 TravelUAV 的终点对齐参考系（`target_aligned_local`）与当前机体系不同，不复用旧动作或文本权重。

全部转换集中在 `sr3m/geometry.py`，作图和执行器共用：机体系位移旋回世界、UE 米坐标与 NED 互转。UE 点坐标转 NED 为 `(X-originX, Y-originY, -(Z-originZ))`；位移不减原点。mountain 爬升 `dz>0`，对应 NED Z 位移必须小于零。

策略状态只含相对起点、在起始 yaw 下表达的位置除以 100，以及相对 yaw 的 sin/cos，共 5 维。这个 100 是输入特征缩放，不是单位转换。

## 3. STOP 与动作掩码

N 个原始转移保留 N+1 个观测样本。最后观测 `endpoint=true`、`stop_target=true`，没有下一动作；无动作占位为 NaN，动作损失先选有效样本再算差，避免 NaN 传播。

以三维欧氏距离定义终点 **10m** 半径，取最后一次离开半径之后到末观测的连续段作为 STOP 正样本。起步就在终点附近的返回路线，只要随后离开过半径，起步仍标负。整条路线都在半径内时准备失败，要求核查任务完成语义。

```text
action_valid = transition_valid AND NOT stop_target
```

所有 STOP 正样本屏蔽动作损失与动作指标，掩码只由真值决定，不能由动作幅度或预测 STOP 决定。原始短尾段保留为几何事实，不变成停止动作。STOP 是独立 logit + BCE，pos_weight=1。

成功半径 SR20 = **20m**，训练 STOP 半径 10m 是预留余量。阈值仅在选定 checkpoint 的 val_seen 上校准：沿专家轨迹，取"首次触发 STOP 时距终点 ≤20m 的路线数"最多的阈值（离线 SR 代理），同分取较高阈值；没有任何路线在 20m 内停下则标记未通过。同时报告提前停（首次触发 >20m）、从不触发和返回路线"首次触发早于末尾连续段"的路线数。（10/22 修订：原规则要求观测级 precision≥95% 且零误停，试跑中把阈值压到 0.95 以上，161 条路线只有 23–36 条触发。）

## 4. 模型、视觉缓存与 checkpoint

A=front，B=front+down 简单拼接，C=down；同一 HAD 模块结构（`had/model.py`）与相同种子的初始参数，缺失视角分支置零。没有 AGL、坡度、终点或路线 ID 输入，没有门控和图像增强。

当前（§11，2026-10-27 起）：冻结 DINOv2-B（`facebook/dinov2-base`，固定 revision），RGB 整帧 bicubic/antialias 缩放到 224×224（不裁剪）、ImageNet mean/std，fp32、禁用 TF32；只缓存最后一层 CLS + patch 2×2 网格平均（每视角 3840 维，float16 存储）；512 维投影（HAD 的 Linear + LayerNorm + ReLU）、文本、融合、策略和 STOP 继续训练。v2 主实验用的是 ImageNet ResNet50 V1 的 2048 维 float32 全局特征（224×224 bilinear），代码见提交 `4b0faf9`。

缓存按 prepared 身份、骨干、预处理和完成行数绑定，读取未完成缓存会失败，中断后从已完成行续写。checkpoint 保存可训练模块、词表快照、数据/动作/STOP 协议、骨干标识、种子、训练统计、Git HEAD 与工作区状态。**注意：run 身份包含 git commit，训练进行中不要提交代码，否则已有 run 无法续跑。**

## 5. 地形分组与评测边界

地形类别为 mountain 与 desert（`PROTOCOL["terrain_categories"]`）。8 张地图（L_MTN_A–D、L_DES_A–D）的捕获版本都与保存回执匹配，地形导出与连续 AGL 证书绑定（两套 `terrain_model.py` 除地图前缀外逐字相同）。AGL = 当前位置 Z − XY 下最高有效地形面（不用屋顶替代），坡度来自该面梯度。低层 <30m，中层 30–80m，高层 >80m。非地形类别的 AGL/坡度保留 NaN，不填零。

随地形/对照是指令需求分组（中文含"随沿途地形调整高度"），不等同于世界 Z 是否变化。两类对照性质不同：train 中 mountain 对照 57 条全在平地定高（每步 |dz| 0.26cm、坡度 0.3°）；desert 对照 30 条在沙丘上定高（每步 |dz| 0、坡度 17.0°），且只有中/高层。随地形组每步 |dz| / 坡度：mountain 39.2cm / 18.8°，desert 36.9cm / 17.5°。

| split | 类别 | 随地形 / 对照 | 低 / 中 / 高 AGL（随地形/对照） |
|---|---|---:|---|
| train | mountain | 73 / 57 | 31/34、26/20、16/3 |
| | desert | 100 / 30 | 63/0、26/22、11/8 |
| val_seen | mountain | 15 / 8 | 8/4、3/4、4/0 |
| | desert | 17 / 6 | 14/0、2/3、1/3 |
| val_unseen | mountain | 37 / 11 | 17/7、13/4、7/0 |
| | desert（L_DES_C） | 37 / 11 | 24/0、9/8、4/3 |

val_unseen 合并为随地形 74 / 对照 22，低/中/高 41/7、22/12、11/3。高 AGL 对照只来自 desert，低层对照只来自 mountain；AGL 匹配 D 仍只对低/中层等权。禁止把训练路线混入独立报告或填造缺失的层。

仅用 train 拟合词表、动作均值/尺度和坡度分箱。动作指标先逐路线后平均。选模分数等权合并：逐分量尺度归一化的平移 RMS、|dyaw|≥1° 的归一化环形 yaw MAE、地形类别（mountain + desert 合并）两 cohort 等权的归一化 dz MAE；尺度 std 下限 0.05/0.05/0.05m 和 1°。评测报告的 `terrain` 为合并结果，`terrain_by_category` 为按类别拆分。

## 6. 离线实验（11/10 决策）

| 日期 | 交付 | 入口 |
|---|---|---|
| 10/21–22 | 英文输入、STOP、地形绑定、分组、过拟合、缓存；A/B 种子 42 试跑（v1，只看 val_seen） | `experiments/prepare.py`、`overfit.py`、`cache.py`、`train.py` |
| 10/22–23 | desert 重建与采集，发布 `benchmark_split_v2`（已完成） | `sr3m/release_v2.py` |
| 10/23–24 | v2 上 prepare、缓存、十路线过拟合（已完成；过拟合 2500 步通过） | `prepare.py`、`cache.py`、`overfit.py` |
| 10/25– | A/B × 三种子训练（本机 3070），冻结 checkpoint/阈值；C 暂不跑 | `train.py` → `selection.json` |
| 11/7–11/9 | val_unseen 分组报告、线性探针、配对区间 | `evaluate.py`、`probe.py`、`summarize.py` |
| 11/10 | 一页研究决策 | `decision/decision.md` |

`python -m experiments.run_all` 按依赖顺序推进并可续跑；训练不读取 val_unseen，val_unseen 只在九次选择全部冻结后打开。

**正式训练**：A/B × seeds 42/43/44（C 暂不跑，`PROTOCOL` 仍保留其定义）。AdamW lr=1e-4、wd=1e-4、batch 32、dropout 0.2；最多 40 epochs、至少 10、patience 5，最小复合 val_seen 分数选模，平分取较早。动作损失为训练 std（带下限）归一化 MSE；yaw 用包角残差，转弯/非转弯按训练频率等权。STOP BCE 权重 0.1，不做类别加权和重采样。

**十路线过拟合**是工程检查：覆盖两个 cohort、三种 AGL 和四类非山地；关闭 dropout、lr=1e-3，要求归一化动作 MSE≤0.05、STOP 最佳 F1≥0.9。其权重不进入正式实验。

**一等交付物**：

1. 恒定直飞 2m、训练均值两种基线与 A/B/C 三种子的总体/转弯指标。
2. mountain 两个 cohort 的 dz、平移、yaw；低/中 AGL 等权分析，高层缺失明确显示。
3. H1/H2/不伤害三项检验（见下方决策规则），以及描述性的 `G = dz_MAE(A) − dz_MAE(B)`、原始 D、AGL 匹配 D、标签幅值、坡度分布与逐路线结果。
4. STOP：首次停止在 20m 内的路线占比、提前停与从不触发路线数、固定阈值、precision/recall/F1 及返回路线提前触发统计。
5. 冻结俯视特征的 AGL/坡度 Ridge 探针：train 拟合，val_seen 从 1e-4…100 选正则，val_unseen 报 MAE/R² 与训练均值基线。

路线 bootstrap 先对三个种子的逐路线指标求平均，再在类别/cohort/AGL 内抽样；区间表示路线不确定性，种子标准差另报。

**决策规则**（2026-10-21 开训前定稿）：

- 背景：train 中对照组 57 条路线全部是平地上定高飞行（每步 |dz| 0.26cm、地形坡度 0.3°），随地形组每步 |dz| 39cm、坡度 18.8°。因此对照组 dz 增益上限约 0.3cm，原来的交互统计 D 几乎等于随地形组增益，不能检验"地形越复杂越有用"；绝对增益随坡度增大也是 |dz| 变大带来的必然结果。判据改为相对量。
- **门槛**：A、B 每个种子的综合分数和转弯 yaw 都须优于两种基线，且三种子平均的配对改善区间下界为正。
- v2 补充：desert 对照组在坡面上定高（坡度 17°、|dz|=0），"不伤害"在 desert 上检验的是"有坡但不应改高度"时俯视是否诱发多余 dz，比 mountain 平地对照更严格。判据本身不变，H1/H2/不伤害均在 mountain + desert 合并路线上计算，按类别拆分只作描述。
- **H1（主判据）**：val_unseen（mountain 与 desert 的留出地图）随地形组相对 dz 改善 `R = 1 − dzMAE(B)/dzMAE(A)`（逐路线 dz MAE 的路线平均之比）。三个种子 R 均 >0，且路线 bootstrap 95% CI 下界 >0。不设最小效应量。
- **H2（支撑）**：坡度分箱为 train 随地形路线有效步的坡度三分位（v2 合并实测 4.25° / 18.57°，三档每步 |dz| 约 10/19/84cm；v1 仅 mountain 时为 1.93° / 21.57°）。统计高坡档 R − 低坡档 R，三种子均 >0 且 CI 下界 >0 为通过。
- **不伤害**：对照组 B 相对 A 的 dz MAE 与平移误差增加量（种子平均）均 ≤ 1cm。
- 结论：门槛未过 → 先处理动作学习；H1 过但伤害对照 → 先查融合；H1+H2 → 核心假设成立，继续主线；仅 H1 → 继续主线，主张限定为"地形跟随受益于俯视"；H1 未过且俯视探针不可读 → 先查深度表示；否则 → 转 benchmark。
- 原始 D、AGL 匹配 D、相对改善与各层结果保留为描述性报告。C 未过关仅作诊断。线性探针失败不能写成"俯视图没有信息"。此结果只是每个地形类别各一张留出地图（L_MTN_C、L_DES_C）的首轮证据。

## 7. 闭环与 desert

闭环不阻塞离线决策。`sr3m/sim.py` 提供执行器，`experiments/closed_loop.py` 运行专家回放门槛和模型飞行；单元测试不替代真实 UE 验收。

**仿真准备**：用 UE 5.8.4 从源码构建 Cosys-AirSim 插件（预编译 5.8.2 不直接用），启用 Python API；MCP 不是依赖。复制 `configs/cosys_settings.example.json` 到实验设置位置，核对 front/down 相机 512×512、90° FOV、曝光和朝向与捕获一致；gimbal 固定 pitch/roll、yaw 跟随机体，用 `simGetCameraInfo` 和真实图像验收，尤其俯视 −90°。Python 环境安装 `cosysairsim`。手动打开路线对应的完整 map package、初始化到起点、记录 AirSim 原点在 UE 世界中的米坐标；运行时显式提供 `--map-package` 和 `--origin-ue-m`。

**执行与门槛**：16 条路线在 `prepared/closed_loop_routes.json` 按几何固定（低/中 AGL × 两 cohort 各 3 条；高 AGL 随地形 2 条；非山地直行、返回各 1 条），属于 val_seen 的工程检查。执行时先按 yaw_t 把机体系动作旋回世界并固定目标点，保持 yaw_t 用 MaxDegreeOfFreedom 平移，到位后 rotateToYaw 到 yaw_t+dyaw，再 hover/取图。专家策略从实际当前状态指向记录的下一世界位姿，通过同一执行器；每步位置误差≤0.5m、yaw≤0.05°，旋转 API margin=0.01°（避免 2° 死区吞掉小转弯）。16/16 完成、零碰撞/超时后模型才能运行；禁止逐步 teleport 冒充回放。模型 STOP 阈值只读 val_seen 冻结选择，不合格则不启动模型闭环。

```bash
python -m experiments.closed_loop --prepared runs/terrain_v1/prepared \
  --episode '<closed_loop_routes.json 中的 episode_id>' \
  --map-package '<对应完整 map package>' --origin-ue-m X Y Z \
  --gate runs/terrain_v1/expert_gate.json --out runs/terrain_v1/expert_route.json
# 模型飞行另加 --checkpoint runs/terrain_v1/models/B_seed42/best.pt
```

成功率用 SR20，另报路线进度、碰撞、超时和提前 STOP，标明返回型路线的解释边界。

**Desert（10/23 已采集并发布 v2，以下为当时的要求）**：沙丘生成方法由数据集侧自行确定（`sr3m/desert_terrain.py` 只是早期的四档起伏预览，不再是定稿方案）。要求：4 张图用不同沙丘场，每张图内都有平缓到陡峭的区域；每张图同时包含"随沿途地形调整高度"路线和在起伏上定高的路线（AGL 中/高层）；复用 mountain 的 AGL 分层、连续离地检查与采集流程。为接入 prepare，每张图需提供与 mountain 相同的地形产物：`scripts/terrain_model.py`、`evidence/persisted-<图ID>.json`、`evidence/continuous-agl-<图ID>.json`、`design/<图ID>.final.json`（含 `route_id`、`instruction_zh`、`height_layer`）。新数据发布为 `benchmark_split_v2`，不原地覆盖 v1；旧 desert 捕获保留不删。留出地图只进 val_unseen，不参与词表、统计、超参、checkpoint 或阈值选择。

## 8. 代码来源

- `had/vendor/{encoders,fusion,policy_head}.py`：HAD-UAV-VLN commit `f01440a` 的逐字节副本，哈希见 `had/vendor/SOURCE.json`。
- `had/model.py`：SR3M `vln_benchmark/unified/had.py` 中 `HadPolicy` 的 dual_relative/concat 路径与原 `TerrainHAD` 合并，模块构建顺序不变。
- `had/tokenizer.py`、`sr3m/data.py`、`sr3m/geometry.py`：来自 SR3M `vln_benchmark`（`unified/tokenizer.py`、`dataset.py`、`unified/data.py`、`unified/contract.py`），只删去未使用的部分。
- 迁移后与迁移前逐位一致：A/B/C 初始化、前向输出、损失、val_seen 基线、prepare 输出（2026-10-21 校验）。
- 旧中文单种子 front/concat/gate 的历史 test 误差 6.83/6.95/6.31 cm 包含旧 desert、协议不同，不能混入本轮结论。

## 9. v3 开发（方向 A，2026-10-26 已收束）

v2 决策后曾尝试 v3：`PROTOCOL` = `sr3m_had_terrain_v3`，加入 Depth Anything V2 Small 相对逆深度（16×16，每视角 256 维）作为 AD/BD 变体的线性输入，并用留一图开发折（留出 L_MTN_x + L_DES_x 的 train + val_seen 路线）比较。单折最小对照（留出 L_MTN_B + L_DES_B，B vs BD 各 seed 42）中 BD 只多改善 2–3%（CI 跨 0），未做完整 v3；之后的度量深度检查也表明现成单目度量深度给不出 AGL。结论与数据见 `runs/terrain_v2/decision/v3_fold_B.md`、`m1_results.md`。

代码已恢复到 v2（与 `d7ba299` 的代码逐字节相同，`PROTOCOL` = `sr3m_had_terrain_v2`）。v3 的实现（`experiments/depth.py`、`fold.py`、`fold_report.py` 与 AD/BD 变体）保留在提交 `a62ad92`，需要时从历史恢复。

## 10. 决策后的诊断变体（2026-10-26）

只作描述，不改变 10/25 的决策，也不进入 H1/H2 判据。定义放在 `PROTOCOL` 之外（`experiments/protocol.py` 的 `DIAGNOSTIC_VARIANTS`，与 `MAIN_VARIANTS` 同理），所以 v2 准备集、特征缓存和已有 checkpoint 保持有效。训练超参、选择规则与 A/B 完全相同，用 `python -m experiments.train --variants <V> --out runs/terrain_v2/models_<v>` 训练到单独目录。

- **C**（仅俯视）：`PROTOCOL` 中原有定义，3 个种子已完成，见 `runs/terrain_v2/decision/c_results.md`。
- **N**（无视觉输入）：只用指令和 5 维状态，两个视角都以零向量输入（与 A、C 对缺失视角的处理相同）。用来检验离线导航指标在不看图时能做到多好，以及 dz 的"不看图"下限。

## 11. DINOv2 重做（2026-10-27 开训前定稿）

**动机**：E1 留一图探针（`runs/terrain_v2/decision/e1_results.md`，只用 train/val_seen 的地图，未读 val_unseen）显示，ResNet 全局特征在未见地图上几乎读不出 dz（0.9%），DINOv2-B 为 16.9%，其中只看俯视就有 13.9%。v2 中"俯视单独不比前视好"和 H2 未通过，可能是弱编码器造成的。本实验只更换冻结视觉特征，重新回答核心问题。

**唯一改动**（`PROTOCOL` = `sr3m_had_terrain_v2_dinov2`）：

- 骨干：DINOv2-B（`facebook/dinov2-base@f9e44c814b77203eaa57a6bdbbd535f21ede1415`），RGB 整帧 bicubic/antialias 缩放到 224×224，不裁剪（保留坡度依赖的画面布局），ImageNet mean/std，fp32、禁用 TF32，冻结。
- 特征：最后一层输出（含最终 LayerNorm）的 CLS + patch token 2×2 网格平均，每视角 3840 维；缓存为 float16（相对舍入误差约 4e-4）。与 E1 探针的特征相关 0.99998（探针用 fp16 计算）。
- 每视角投影仍为 HAD 的 Linear + LayerNorm + ReLU（3840→512）。骨干在所有可训练模块之后构建，同一种子下 A/B/C/N 的初始参数相同。
- 不变：v2 划分与准备（词表、动作统计、坡度分箱只用 train；数据相同，结果与 v2 一致）、超参（AdamW lr 1e-4、wd 1e-4、batch 32、dropout 0.2、10–40 epochs、patience 5）、选模分数、STOP 校准、种子 42/43/44。不根据任何结果调参。
- 不加相对深度：探针中 DINOv2 + 深度最好（20.4%），但 v3 单折显示深度在策略层只多 2–3%，而且会多一个模型依赖；本实验只检验"换编码器"。

**运行顺序**：prepare → 十路线过拟合（§6 标准）→ 缓存 → A/B × 42/43/44（主实验）→ 6 个 run 全部在 val_seen 冻结后打开 val_unseen → C × 42/43/44 与 N × 42（描述性，在 A/B 冻结之后训练）。输出 `runs/terrain_v2_dino/`，缓存 `~/sr3mxhad_runs/terrain_v2_dino/features`。需要 `PYTHONPATH=~/sr3mxhad_runs/pylib` 提供 transformers（prepare、train、evaluate 不需要；cache、overfit、closed_loop 需要）。

**val_unseen 的使用**：v2 时已经打开过一次（ResNet 模型）。DINOv2 的选择只依据 train/val_seen 地图上的留一图探针，没有使用 val_unseen。本实验是该划分的第二次查看，所有判据在开训前固定如下，事后不修改。

**主判据**：与 §6 的定义、阈值和决策表完全相同（门槛、H1、H2、不伤害），比较 B_dino 与 A_dino。

**次判据（预先登记）**：

- **S1 更强编码器是否放大俯视的帮助**：Δ = R_dino − R_resnet，R = 1 − dzMAE(B)/dzMAE(A)（随地形组，三种子逐路线平均后求比）。R_resnet 取 v2 已冻结的 val_unseen 报告。两者用同一组路线 bootstrap 抽样（2000 次，按 §6 分层）。Δ > 0 且 95% CI 下界 > 0 → 通过。
- **S2 俯视单独是否优于前视单独**：C_dino 相对 A_dino 的随地形相对 dz 改善，三个种子都 > 0 且 CI 下界 > 0 → 通过（v2 为 0.0%，CI [−3.6, 3.2]）。
- **S3 相对不看图**：A_dino、B_dino、C_dino 相对 N_dino 的随地形 dz 改善，仅描述。
- **绝对水平**：B_dino 与 B_resnet 的随地形 dz 和对照 dz，仅描述。

**结论映射**：

| 结果 | 结论 |
|---|---|
| H1 过 + S1 过 | 更强的编码器放大了俯视的帮助；论文主结果用 DINOv2 版，ResNet 作对照 |
| H1 过、S1 不过 | 俯视的帮助存在，但不随编码器变强而增大；两版都报告 |
| H1 不过 | 更强编码器下俯视的帮助未复现；主结果仍以 v2 为准，并讨论差异 |

H2 与不伤害按 §6 决策表解读；S2 通过时，"俯视单独优于前视"可以写进论文，否则维持 v2 的"帮助来自双视角互补"。

## 12. 融合修复 R1：反事实定高指令（2026-10-27 开训前定稿）

**动机**：`runs/terrain_v2_dino/decision/fusion_diag.md`。DINOv2 A/B 都几乎不用指令里的高度条件（把 FOLLOW 换成 HOLD，train 上 β 只降约 0.07）；B 在未见陡坡定高路线上的误爬升来自俯视（换成均值后消失）。训练集里"坡上定高"只有 30 条 desert 路线，follow/hold 可以靠记路线区分。

**R1**（`train.py --augment hold`，定义在 `PROTOCOL` 之外，run 身份记录 `augment`）：train 中每条地形类随地形路线（desert 100 + mountain 73）再加一份副本，指令里的随地形短语换成定高短语，有效步的 dz 目标设为 0（定高时任何状态下的正确 dz 都是 0），图像特征、水平动作、yaw、STOP 不变。train 行数 127,856 → 151,050。词表、动作统计、坡度分箱不重新拟合；验证集与选模分数不变；仍是简单拼接，不加门控、AGL 输入或图像增强。

**开发集**：只用 val_seen。新增反事实检查：val_seen 地形类随地形路线换成 HOLD 指令（同一替换），真值 dz 恒为 0。

**开发判据**（R1 vs 同种子的 DINOv2 基线，先 B 再 A，种子 42）：

1. 反事实 HOLD：β ≤ 0.05，且每步 |dz| 明显下降（报告绝对值）。
2. 原指令随地形：val_seen dz MAE 变差 ≤ 0.5 cm，β 下降 ≤ 0.02（约为三种子间标准差）。
3. 选模分数变差 ≤ 0.005（约为三种子间标准差）。

三条都满足 → 讨论是否用 R1 跑 A/B × 3 种子并第三次查看 val_unseen（需另行确认）；不满足 → 记录为负结果，再讨论 R2（文本门控）。val_unseen 不参与 R1 的任何开发与选择。
