# SR3M × HAD

核心检验：**mountain 中要求随地形调整高度的路线，是否比对照路线更受益于俯视图，尤其是 dz 误差。**

用户研究方向见 [研究计划](SR3M%20×%20HAD%20研究计划.md)，执行安排见 [离线实验计划](离线实验计划.md) 和 [闭环实验计划](闭环实验计划.md)，准确的数据与坐标定义见 [数据与输入合同](数据与输入合同.md)。

## 当前实现

已实现英文数据准备、旧 desert 过滤、地形绑定与 AGL/坡度、十路线过拟合检查、冻结 CNN 特征缓存、A/B/C × 三种子训练、复合选模、STOP 校准、分组评测、线性探针和配对决策汇总。闭环提供独立执行器与 16 路线专家回放门槛，需真实 Cosys-AirSim 环境才能验证。

| split | 非 desert 路线 | mountain 随地形调整 / 对照 |
|---|---:|---:|
| train | 898 | 73 / 57 |
| val_seen | 161 | 15 / 8 |
| val_unseen | 199 | 37 / 11 |
| 既有 test（本轮不使用） | 147 | 0 / 0 |

全量 mountain 的 125/76 不能当作独立评测规模。val_unseen 高 AGL 没有对照；报告保留缺失，另给低/中 AGL 等权比较。

## 运行

依赖现有 SR3M 代码、PyTorch/torchvision、NumPy、Pillow、Matplotlib 和 pytest。当前可用环境是 `/home/qlj/h3c_pro/HAD-UAV-VLN/.venv`；默认读取相邻 `../SR3M`，其他部署位置设置 `SR3M_SOURCE`。官方骨干文件默认在 `~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth`。

```bash
export SR3MXHAD_PYTHON=/home/qlj/h3c_pro/HAD-UAV-VLN/.venv/bin/python
"$SR3MXHAD_PYTHON" -B -m pytest -q -p no:cacheprovider tests
"$SR3MXHAD_PYTHON" -u -B run_experiments.py
```

调度器先完成 10 路线检查，再缓存并训练；每组最多 30 epochs，至少 10，patience 5。中断后重跑同一命令，缓存从已完成行续读，训练从最后完整 epoch 恢复。数据与模型输出不进 Git。

产物位置：

- `runs/terrain_v1/pipeline.json`：当前阶段、完成或失败状态；对应 `*.log` 记录过程。
- `runs/terrain_v1/prepared/`：1,330 项英文词表、训练统计、地形标签和固定 16 路线。
- `/home/qlj/sr3mxhad_runs/terrain_v1/features/`：front/down 2048 维 float32 缓存。
- `runs/terrain_v1/models/{A,B,C}_seed{42,43,44}/`：best/last、训练曲线和 val_seen 选择记录。
- `runs/terrain_v1/val_unseen/`、`probe/`、`decision/decision.md`：冻结后的主评测、探针和一页决策报告。

需要其他数据或缓存位置时使用 `run_experiments.py --root ... --terrain ... --cache ... --out ...`。云端复用 `/root/autodl-tmp/sr3m-openvla` 时，将 `SR3M_SOURCE` 指向该目录，并指定对应数据根目录；不要把旧中文模型当作本轮模型。

## 其他入口

`plot_actions.py` 绘制五条非 desert 专家路线、STOP 后缀及无动作末观测，并检查爬升到 NED 的符号。它显示专家几何标签，不是模型结果。

`desert_terrain.py --out runs/desert_preview` 生成四档共享布局的程序化地形 OBJ、量化指标和预览。它们是 Blender 可导入的离线几何，不是已建好的 UE 地图。

已有中文单种子 front/concat/gate 的历史 test 误差为 6.83/6.95/6.31 cm；这些实验包含旧 desert，协议不同，不能混入本轮英文三种子结论。
