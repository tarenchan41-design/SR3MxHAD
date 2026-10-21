# SR3M × HAD 项目事实与固定约定

- 数据划分固定为已有 `benchmark_split_v1`，精确归属以 `/mnt/f/WSL_ColdStorage/datasets/DatasetCaptureBatches/workbench/benchmark_split_v1/{train,val_seen,val_unseen,test}.jsonl` 为准。轨迹数分别为 1027、184、199、196；划分种子为 20260929。不得重新随机划分或为地形、语言、模型对照移动轨迹。
- 地图明细、动作公式、输入边界和核查结果见 [数据与输入合同.md](数据与输入合同.md)。后续地形档位和英文译文作为相同 episode ID 的附加属性，沿用冻结归属。
- 当前已训练模型的动作合同是 `capture_adjacent_yaw_fru_m_rad_v1`：原始相邻记录，当前 yaw 的前/右/上，米/弧度；状态在初始 yaw 坐标系表达。复用 `../SR3M/vln_benchmark/unified/{contract,data,had,tokenizer}.py`。旧 TravelUAV 的 `target_aligned_local` 不能直接用于本合同。
- 主实验语言使用原始中文；原文保留目标编号等标识。英文版若新增，保留中文并另存译文，翻译训练与评测保持语言条件一致，不直接交给现有中文 checkpoint。
- 当前没有真实 AGL 输入。记录终点辅助标签不等于导航 STOP；动作还原图展示专家标签，不是模型预测或闭环轨迹。
- [plot_actions.py](plot_actions.py) 复用上述实际读取器，检查源位姿与动作并输出五条训练轨迹的 PNG/PDF；不写原始数据、划分或训练目录。运行方式与验证边界见数据合同。
