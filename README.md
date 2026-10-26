# SR3M × HAD

核心检验：**地形场景（mountain + 重建 desert）中要求随地形调整高度的路线，是否比对照路线更受益于俯视图，尤其是 dz 误差。**

- 研究动机、投稿目标、时间线：[docs/research_plan.md](docs/research_plan.md)
- 数据、坐标、STOP、评测与决策规则：[docs/protocol.md](docs/protocol.md)
- 仓库整合的长期计划（对账、数据发布等）：[docs/AGENT_TASK_architecture.md](docs/AGENT_TASK_architecture.md)

## 代码结构

```
sr3m/          benchmark，将来随数据集公开；不得 import had / experiments
  data.py        发布划分与捕获文件的只读读取、校验
  release_v2.py  发布 benchmark_split_v2（v1 去旧 desert + 新 L_DES 四图）
  geometry.py    动作合同、FRU/UE/NED 坐标转换
  sim.py         Cosys-AirSim 执行器（先平移后转 yaw）
  desert_terrain.py  早期程序化沙丘预览（非定稿）
had/           方法；不得 import sr3m / experiments
  model.py       TerrainHAD：A/B/C 共用的 HAD 双视角策略
  tokenizer.py   词表
  vendor/        HAD-UAV-VLN@f01440a 原始模块，逐字节副本，不改
experiments/   唯一同时使用两者的地方
  protocol.py    PROTOCOL 常量、prepared 数据接口、路径配置
  run_all.py     完整流水线：prepare → overfit → cache → train → evaluate → probe → summarize
  metrics.py  closed_loop.py  plot_actions.py  及各阶段脚本
configs/       paths.example.toml（复制为 paths.toml）、Cosys-AirSim 设置样例
tests/         单元测试，含包边界测试
```

## 运行

依赖：Python ≥ 3.11、PyTorch/torchvision、NumPy、Pillow、Matplotlib、pytest。本机环境是 `/home/qlj/h3c_pro/HAD-UAV-VLN/.venv`；ResNet50 权重默认读 `~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth`。所有命令在仓库根目录执行。

```bash
cp configs/paths.example.toml configs/paths.toml   # 首次，按本机修改
export PY=/home/qlj/h3c_pro/HAD-UAV-VLN/.venv/bin/python
$PY -m pytest -q
$PY -u -m experiments.run_all
```

流水线可中断续跑：缓存从已完成行续读，训练从最后完整 epoch 恢复；状态见 `runs/terrain_v2/pipeline.json` 与各阶段 `*.log`。训练进行中不要提交代码（run 身份绑定 git commit）。

产物（均不进 Git）：

- `runs/terrain_v2/prepared/`：词表、训练统计、地形标签、16 条闭环路线、val_seen 基线
- `cache_dir`（见 paths.toml）：front/down 2048 维 float32 特征
- `runs/terrain_v2/models/{A,B,C}_seed{42,43,44}/`：best/last、训练曲线、val_seen 选择
- `runs/terrain_v2/val_unseen/`、`probe/`、`decision/decision.md`：冻结后的主评测、探针、决策报告

其他入口：`python -m experiments.plot_actions`（五条专家轨迹的动作重建图，输出 `figures/`）、`python -m sr3m.desert_terrain --out runs/desert_preview`、`python -m experiments.closed_loop`（见 protocol.md §7）。
