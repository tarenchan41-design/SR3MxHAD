# SR3M × HAD

以地形为核心组织 SR3M benchmark，检验 HAD 双视角融合在不同地形条件下的收益。研究方向见 [研究计划](SR3M%20%C3%97%20HAD%20%E7%A0%94%E7%A9%B6%E8%AE%A1%E5%88%92.md)。

核心问题：**地形越复杂，加入俯视图带来的收益是否越大？** 这是待验证的假设，报告中的“实验发现”目前不能作为本项目已得到的结论。

固定划分、动作坐标合同、全量核查及 instruction 语言决定见 [数据与输入合同](数据与输入合同.md)；五条专家轨迹的动作图见 PNG（`figures/actions_5_trajectories.png`） / 5 页 PDF（`figures/actions_5_trajectories.pdf`）。项目后续执行约定见 [AGENTS.md](AGENTS.md)。

## 仓库范围与运行依赖

本仓库保存本目录的源码与文档；数据集、模型权重、实验输出、缓存和生成图表不随仓库上传。

`plot_actions.py` 依赖相邻目录 `../SR3M` 中的 `vln_benchmark` 代码、已准备的 tokenizer 及本地数据集；仅克隆本仓库还不能直接运行。下文指向 `../SR3M` 或绝对路径的内容均为原工作区的来源说明，不是仓库内文件。`figures/` 中的 PNG/PDF 由脚本在满足这些依赖后生成。

## 已有基础

2026-10-20 核对了相邻项目的代码、冻结数据索引与正式实验输出。可复用的训练与评测代码仍位于 SR3M/vln_benchmark（`../SR3M/vln_benchmark`），本目录新增固定数据合同与动作绘图入口。

| 内容 | 已核对的状态 | 入口 |
| --- | --- | --- |
| 数据与划分 | 1,606 条轨迹、32 张地图、195,594 帧；四个 split 的 episode ID 无交集 | split_config.json（`../SR3M/vln_benchmark/split_config.json`） |
| 坐标与动作 | 已实现相邻位姿到局部连续动作的转换 | contract.py（`../SR3M/vln_benchmark/unified/contract.py`） |
| 数据接入 | 已有 front/down 图像、中文指令与相对位姿读取器 | data.py（`../SR3M/vln_benchmark/unified/data.py`） |
| HAD 训练 | 已有前视、双视角拼接、双视角门控配置和训练入口 | configs（`../SR3M/vln_benchmark/unified/configs`）、train.py（`../SR3M/vln_benchmark/unified/train.py`） |
| 离线评测 | 已有动作误差、逐路线/地图、转弯/升降分组及异常率 | metrics.py（`../SR3M/vln_benchmark/unified/metrics.py`） |
| 地形查询 | 注册表共 32 张地图，其中 12 张记录了地形导出来源，但绑定状态仍未确认 | scene_sources.json（`../SR3M/vln_benchmark/unified/scene_sources.json`）、terrain.py（`../SR3M/vln_benchmark/unified/terrain.py`） |

数据根目录：`/mnt/f/WSL_ColdStorage/datasets/DatasetCaptureBatches/workbench`。现有冻结划分为 `benchmark_split_v1`：

| split | 轨迹数 | 原始相邻转移数 |
| --- | ---: | ---: |
| train | 1,027 | 125,481 |
| val_seen | 184 | 21,775 |
| val_unseen | 199 | 22,951 |
| test | 196 | 23,781 |

已有正式 test 输出如下，均为同一批 23,781 个转移，异常率为 0。数值从原始 `report.json` 读取；这是已有单种子离线结果，本次没有重新训练。

| 方法 | 平移平均误差（cm）↓ | 环形 yaw MAE（°）↓ |
| --- | ---: | ---: |
| HAD 前视 | 6.83 | 0.669 |
| HAD 双视角拼接 | 6.95 | 0.710 |
| HAD 双视角门控 | 6.31 | 0.664 |

原始报告：前视（`../SR3M/vln_benchmark/runs/had-formal-20261010/results/unified/had-formal-20261010-had_front-test/report.json`）、拼接（`../SR3M/vln_benchmark/runs/had-formal-20261010/results/unified/had-formal-20261010-had_dual_concat-test/report.json`）、门控（`../SR3M/vln_benchmark/runs/had-formal-20261010/results/unified/had-formal-20261010-had_dual_gate-test/report.json`）。解释与边界见 10/13 实验结果（`../SR3M/01_HAD与OpenVLA_本轮实验结果.md`）。旧 `unified/README.md` 开头仍停留在正式训练前，不能据此判断目前没有正式结果。

## 整合时需要修正的五点

1. **已有适配不重做。** 研究计划第一周的坐标转换和数据接入已有实现。先核对现有接口，再补地形分析与实验缺口。
2. **记录终点与导航 STOP 分开。** 当前动作是 `[dx, dy, dz, dyaw]`，单位为米/弧度，坐标轴为当前朝向的前/右/上；输入状态使用起始朝向坐标系。辅助头预测的是记录终点，不能直接解释为成功停止。连续 yaw 使用环形 MAE；如需准确率，应先定义角度容差或类别。
3. **地形难度与动作难度分开。** 轨迹 Z 变化、转弯和升降可以直接统计，但不能代替地表起伏与坡度。先核对地形来源、地图版本、坐标和覆盖范围；无可靠地形的轨迹标记为缺失。难度分数及易/中/难阈值只在 train 上确定，之后固定。完整轨迹统计仅供分组分析，不作为单步策略输入。
4. **对照组尚不完整。** 已有 front、dual-concat、dual-gate，没有仅俯视 profile；当前门控配置也未启用真实 AGL。新增仅俯视组并对齐训练条件，门控与拼接分别报告。双视角参数量更多，若要把收益归因于视角信息，还需容量控制；“可机载部署”需用目标设备的延迟与显存测量支持。
5. **已有 test 不能反复用于开发。** 先在 train/validation 开发分档和模型，保留现有冻结划分及历史结果。已有 test 可作描述性分析；若据此选择方案，需要另留未使用的数据确认结论。多种子比较以同一批路线配对，报告各档增益与波动；现有输出没有闭环 SR/SPL。

## 实施顺序与完成判据

| 顺序 | 工作 | 完成判据 |
| --- | --- | --- |
| 1 | 复用动作合同，抽查真实数据 | 至少 5 条轨迹的相邻动作可还原到下一位姿；确认前/右/上、yaw 跨界和末帧语义，双视图文件可读 |
| 2 | 核对地形并定义难度 | 给出地形覆盖率、缺失数、评分公式，以及每个 split 的分档路线数；能区分地表难度与轨迹运动难度 |
| 3 | 扩展地形分组评测 | 在 validation 上复用已有预测，按路线配对比较 front/concat/gate；转弯、升降组同时保留 |
| 4 | 补仅俯视与多种子 | front/down/dual 使用相同划分、状态、训练预算和选模规则；各组至少 3 个种子，附固定前进和训练均值基线 |
| 5 | 决定主线并接闭环 | 用 validation 判断是否值得继续，冻结方案后做保留数据评测；闭环再验证动作执行、碰撞、停止与成功率 |

离线阶段对误差指标定义每档增益 `G = error_front - error_dual`，正值表示双视角更好。核心比较是 `G_hard - G_easy`，同时报告绝对值、相对改善、路线数与不确定性，检查结果是否被地图类别或运动难度解释。3 个种子用于检查稳定性，不能仅凭均值排序宣称统计显著。

现阶段优先推进第 1–3 项，用已有模型和预测判断新增实验的价值。生成式外观增强保留为研究计划中的支线；投稿日期和软件兼容版本在实际使用前另查官方来源。

## 本次验证

冻结索引统计与 `prepared_v2/statistics.json` 的 train 统计一致；三份正式 HAD test 报告均包含 23,781 个有效转移，且 `closed_loop=false`、`SR=null`、`SPL=null`。这不代表本次重新读取了全部图像或复跑了模型。

已有动作合同与指标测试共 6 项通过。复现命令：

```bash
cd /mnt/e/DL/Transformers/Projects/SR3M
/home/qlj/h3c_pro/HAD-UAV-VLN/.venv/bin/python -B -m pytest -q -p no:cacheprovider \
  vln_benchmark/unified/tests/test_contract.py \
  vln_benchmark/unified/tests/test_metrics.py
```
