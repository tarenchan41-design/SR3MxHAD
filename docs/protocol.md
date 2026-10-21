# SR3M × HAD 实验协议

本文件是实验的执行合同：数据、动作坐标、STOP、缓存、评测、11/3 离线决策和闭环流程。研究动机与时间线见 [research_plan.md](research_plan.md)。代码中的常量以 `experiments/protocol.py` 的 `PROTOCOL` 为准；改动 `PROTOCOL` 会使已有 prepared 数据和特征缓存失效。

## 1. 实验数据与语言

原始发布划分为 1027/184/199/196 条。主实验只创建过滤视图，不改原始 JSONL、不重分配路线；旧 desert 全部排除后为 train/val_seen/val_unseen/test = 898/161/199/147 条。原始数据和地形导出的位置见 `configs/paths.toml`，均只读。

英文译文在捕获文件和索引的 `instruction`，中文原文在 `instruction_zh`。读取器验证索引、捕获与 source-plan 的一致性；分组使用中文原文，策略使用英文。仅用 898 条训练指令拟合词表，实测 1,330 项、上限 256 tokens，三集合均无截断。val_seen/unseen 未知 token 占比约 0.673%/2.112%，不反向扩充词表。

| 集合 | 观测 | 原始转移 | STOP 正样本（含末观测） | 有效动作 |
|---|---:|---:|---:|---:|
| train | 111397 | 110499 | 5390 | 106007 |
| val_seen | 19348 | 19187 | 966 | 18382 |
| val_unseen | 23150 | 22951 | 1196 | 21954 |

prepare 不读取既有 test 的捕获与动作，只在发布索引身份检查中包含其 JSONL。test 不参与本轮选模与开发；重建 desert 承担新的未见地形外测。

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

成功半径 SR20 = **20m**，训练 STOP 半径 10m 是预留余量。阈值仅在选定 checkpoint 的 val_seen 上校准：非空预测 precision≥95%，专家观测序列首次触发 STOP 时无距终点 >20m 的误停，再最大化召回；没有合格阈值则标记未通过。另报告返回路线的"首次触发早于末尾连续段"，不藏进 SR20。

## 4. 模型、视觉缓存与 checkpoint

A=front，B=front+down 简单拼接，C=down；同一 HAD 模块结构（`had/model.py`）与相同种子的初始参数，缺失视角分支置零。没有 AGL、坡度、终点或路线 ID 输入，没有门控和图像增强。

固定 ImageNet ResNet50 V1、RGB、224×224 bilinear/antialias、ImageNet mean/std；冻结 CNN 和 BN，禁用 TF32。只缓存 CNN 的 2048 维 float32 输出；512 维投影、文本、融合、策略和 STOP 继续训练。

缓存按 prepared 身份、骨干、预处理和完成行数绑定，读取未完成缓存会失败，中断后从已完成行续写。checkpoint 保存可训练模块、词表快照、数据/动作/STOP 协议、骨干标识、种子、训练统计、Git HEAD 与工作区状态。**注意：run 身份包含 git commit，训练进行中不要提交代码，否则已有 run 无法续跑。**

## 5. 地形分组与评测边界

四张 mountain 地图的捕获版本已与保存回执匹配，地形导出与连续 AGL 证书绑定。AGL = 当前位置 Z − XY 下最高有效地形面（不用屋顶替代），坡度来自该面梯度。低层 <30m，中层 30–80m，高层 >80m。非 mountain 的 AGL/坡度保留 NaN，不填零。

125/76 是指令需求分组（中文含"随沿途地形调整高度"），不等同于世界 Z 是否变化。

| split | mountain 随地形 / 对照 | 低 / 中 / 高 AGL（随地形/对照） |
|---|---:|---|
| train | 73 / 57 | 31/34、26/20、16/3 |
| val_seen | 15 / 8 | 8/4、3/4、4/0 |
| val_unseen | 37 / 11 | 17/7、13/4、7/0 |

全量 125/76 不能当作独立评测规模。val_unseen 高 AGL 无对照，报告保留缺失，低/中层等权比较；禁止把训练路线混入独立报告或填造高层对照。

仅用过滤后的 train 拟合词表、动作均值/尺度和坡度分箱。动作指标先逐路线后平均。选模分数等权合并：逐分量尺度归一化的平移 RMS、|dyaw|≥1° 的归一化环形 yaw MAE、mountain 两 cohort 等权的归一化 dz MAE；尺度 std 下限 0.05/0.05/0.05m 和 1°。

## 6. 离线实验（11/3 前）

| 日期 | 交付 | 入口 |
|---|---|---|
| 10/21–23 | 英文输入、STOP、地图/地形绑定、分组、16 路线固定 | `experiments/prepare.py` |
| 10/24–26 | 十路线过拟合、缓存一致性、基线与 epoch 计时 | `overfit.py`、`cache.py` |
| 10/27–29 | A/B/C 首个种子，只看 val_seen | `train.py` |
| 10/30–11/1 | 其余六次训练，冻结 checkpoint/阈值 | `train.py` → `selection.json` |
| 11/2 | val_unseen 分组报告、线性探针、配对区间 | `evaluate.py`、`probe.py`、`summarize.py` |
| 11/3 | 一页研究决策 | `decision/decision.md` |

`python -m experiments.run_all` 按依赖顺序推进并可续跑；训练不读取 val_unseen，val_unseen 只在九次选择全部冻结后打开。

**九次训练**：A/B/C × seeds 42/43/44。AdamW lr=1e-4、wd=1e-4、batch 32、dropout 0.2；最多 30 epochs、至少 10、patience 5，最小复合 val_seen 分数选模，平分取较早。动作损失为训练 std（带下限）归一化 MSE；yaw 用包角残差，转弯/非转弯按训练频率等权。STOP BCE 权重 0.1，不做类别加权和重采样。

**十路线过拟合**是工程检查：覆盖两个 cohort、三种 AGL 和四类非山地；关闭 dropout、lr=1e-3，要求归一化动作 MSE≤0.05、STOP 最佳 F1≥0.9。其权重不进入正式实验。

**一等交付物**：

1. 恒定直飞 2m、训练均值两种基线与 A/B/C 三种子的总体/转弯指标。
2. mountain 两个 cohort 的 dz、平移、yaw；低/中 AGL 等权分析，高层缺失明确显示。
3. H1/H2/不伤害三项检验（见下方决策规则），以及描述性的 `G = dz_MAE(A) − dz_MAE(B)`、原始 D、AGL 匹配 D、标签幅值、坡度分布与逐路线结果。
4. STOP precision/recall/F1、固定阈值、首次误停距离及返回路线提前触发统计。
5. 冻结俯视特征的 AGL/坡度 Ridge 探针：train 拟合，val_seen 从 1e-4…100 选正则，val_unseen 报 MAE/R² 与训练均值基线。

路线 bootstrap 先对三个种子的逐路线指标求平均，再在类别/cohort/AGL 内抽样；区间表示路线不确定性，种子标准差另报。

**决策规则**（2026-10-21 开训前定稿）：

- 背景：train 中对照组 57 条路线全部是平地上定高飞行（每步 |dz| 0.26cm、地形坡度 0.3°），随地形组每步 |dz| 39cm、坡度 18.8°。因此对照组 dz 增益上限约 0.3cm，原来的交互统计 D 几乎等于随地形组增益，不能检验"地形越复杂越有用"；绝对增益随坡度增大也是 |dz| 变大带来的必然结果。判据改为相对量。
- **门槛**：A、B 每个种子的综合分数和转弯 yaw 都须优于两种基线，且三种子平均的配对改善区间下界为正。
- **H1（主判据）**：val_unseen 随地形组相对 dz 改善 `R = 1 − dzMAE(B)/dzMAE(A)`（逐路线 dz MAE 的路线平均之比）。三个种子 R 均 >0，且路线 bootstrap 95% CI 下界 >0。不设最小效应量。
- **H2（支撑）**：坡度分箱为 train 随地形路线有效步的坡度三分位（实测 1.93° / 21.57°，三档每步 |dz| 约 8/15/94cm）。统计高坡档 R − 低坡档 R，三种子均 >0 且 CI 下界 >0 为通过。
- **不伤害**：对照组 B 相对 A 的 dz MAE 与平移误差增加量（种子平均）均 ≤ 1cm。
- 结论：门槛未过 → 先处理动作学习；H1 过但伤害对照 → 先查融合；H1+H2 → 核心假设成立，继续主线；仅 H1 → 继续主线，主张限定为"地形跟随受益于俯视"；H1 未过且俯视探针不可读 → 先查深度表示；否则 → 转 benchmark。
- 原始 D、AGL 匹配 D、相对改善与各层结果保留为描述性报告。C 未过关仅作诊断。线性探针失败不能写成"俯视图没有信息"。此结果只是单张留出 mountain 地图的首轮证据。

## 7. 闭环（11/4 后）与 desert

闭环不阻塞 11/3 决策。`sr3m/sim.py` 提供执行器，`experiments/closed_loop.py` 运行专家回放门槛和模型飞行；单元测试不替代真实 UE 验收。

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

**Desert（11/10 前）**：`python -m sr3m.desert_terrain --out runs/desert_preview` 用种子 20261029、1000m 区域、257×257 网格生成共享基础沙丘场，四档高程 std 0/10/25/50m，最大坡度约 0/6.77/16.53/30.68°；输出 OBJ、heightfields 和预览，仅为离线几何。后续：Blender 建模、材质/地标控制、UE 导入、AGL 路线与指令；四档保持材质、照明、相机、AGL 与任务分布可比。四图全部为未见地形测试，不能用于词表、归一化、超参、checkpoint 或阈值选择。

## 8. 代码来源

- `had/vendor/{encoders,fusion,policy_head}.py`：HAD-UAV-VLN commit `f01440a` 的逐字节副本，哈希见 `had/vendor/SOURCE.json`。
- `had/model.py`：SR3M `vln_benchmark/unified/had.py` 中 `HadPolicy` 的 dual_relative/concat 路径与原 `TerrainHAD` 合并，模块构建顺序不变。
- `had/tokenizer.py`、`sr3m/data.py`、`sr3m/geometry.py`：来自 SR3M `vln_benchmark`（`unified/tokenizer.py`、`dataset.py`、`unified/data.py`、`unified/contract.py`），只删去未使用的部分。
- 迁移后与迁移前逐位一致：A/B/C 初始化、前向输出、损失、val_seen 基线、prepare 输出（2026-10-21 校验）。
- 旧中文单种子 front/concat/gate 的历史 test 误差 6.83/6.95/6.31 cm 包含旧 desert、协议不同，不能混入本轮结论。
