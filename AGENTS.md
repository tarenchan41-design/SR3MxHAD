# SR3M × HAD 项目事实与执行约定

- 研究目标以用户维护的 [研究计划](SR3M%20×%20HAD%20研究计划.md) 为准。11/3 前优先检验 mountain 两个 cohort 的双视角增益差异；工程整合不能取代这一交付。
- 当前执行合同见 [数据与输入合同](数据与输入合同.md)、[离线实验计划](离线实验计划.md) 和 [闭环实验计划](闭环实验计划.md)。主实验为英文 A/B/C × seeds 42/43/44，简单拼接，不包含门控、AGL 输入或图像增强。
- 原始 `benchmark_split_v1` 和捕获数据只读。实验过滤全部旧 desert，其他路线归属保持不变；train/val_seen/val_unseen/test 剩余 898/161/199/147 条。不得重抽划分或移动关联路线。
- 仅用过滤后的 train 拟合词表、动作均值、尺度和坡度分箱。val_seen 选择 checkpoint、STOP 阈值及探针正则；冻结九次训练后才运行 val_unseen 主报告。既有 test 不进入当前开发；重建 desert 四图全部外测。
- 原始 `instruction_zh` 保留，125/76 按中文“随沿途地形调整高度”核对。独立 val_unseen 是 37/11，低/中/高 AGL 对照为 17/7、13/4、7/0；禁止把训练路线混入独立报告或填造高层对照。
- 数据、HAD 模块复用 `../SR3M/vln_benchmark`，可用 `SR3M_SOURCE` 指定实际位置。本仓库实现实验协议、缓存、训练、汇总及执行器，不复制旧源码树、不修改旧 checkpoint。
- 动作合同仍为 `capture_adjacent_yaw_fru_m_rad_v1`：当前 yaw 的前/右/上，米/弧度；状态使用起始 yaw 坐标系。转换集中在 `coordinates.py`，作图和执行器共用。旧 `target_aligned_local` 权重不能当作当前动作权重。
- STOP 是独立头；正样本为终点 10m 内的末尾连续段，成功半径 20m。最后一个观测必须存在；全部 STOP 正样本屏蔽动作损失和动作指标，不能由动作幅度或预测 STOP 决定掩码。
- 缓存官方 ImageNet ResNet50 CNN 的 2048 维特征，使用固定预处理、冻结 BN、严格 float32（禁用 TF32）。投影、文本和策略继续训练。缓存身份和完成行数必须匹配，不能读取未完成文件。
- `run_experiments.py` 顺序执行：准备、十路线过拟合、缓存、九次训练、冻结后 val_unseen、线性探针、决策汇总。日志及数据产物在忽略的 `runs/`，特征缓存默认在 Linux 文件系统 `/home/qlj/sr3mxhad_runs/terrain_v1/features`。
- Cosys-AirSim 仅在工程副本手动配置后运行。MCP 不是关键路径。先平移后转 yaw；固定 16 条路线全部专家回放通过后才能运行模型。离线地形预览、执行器单元测试不等于已经完成真实 UE 闭环。
- 交付前检查 STOP 梯度、缓存/原图一致、yaw 单位和顺序、分组/空组、词表/划分身份及报告隔离。不要另建 evidence/、verify_* 或哈希清单；当前 Git HEAD 加工作区状态记录代码版本。
