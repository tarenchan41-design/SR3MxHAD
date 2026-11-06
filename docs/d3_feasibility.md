# D3 真实数据可行性结论（2026-11-05）

研究计划"11/10 之后：D3/D4 落地"第 4 项的产出。核实的问题：UAV-Flow 和 AirNav 有没有俯视或近俯视视角、地形类型如何、位姿和动作能否转成 dx/dy/dz/yaw/stop。

依据：两篇论文原文；AirNav 代码 github.com/nopride03/AirNav @ `1263fce`；UAV-Flow 的 HF 数据（`wangxiangyu0814/UAV-Flow`），通过 datasets-server 抽看了 4 条样本的日志和图像。

## 1. 对照

| | **SR3M v3（我们）** | **UAV-Flow**（NeurIPS 2025 D&B） | **AirNav**（arXiv 2601.03707） |
|---|---|---|---|
| 图像来源 | UE 渲染 | **真实航拍**（DJI Mavic 3T RTK），256² JPEG | **正射影像裁剪**：按位姿从 SensatUrban 正射图做透视变换裁出（`vlnce/observation/cropclient.py`），不是相机拍的 |
| 视角 | 前视 + 俯视 | **只有前视**（看了 4 张样本图，都是水平前视） | **只有俯视**（prompt 写 "top-down UAV view"；视野边长 = 2 × 离地高度，相当于 FOV 90° 俯视） |
| 离地高度 | 20–30 m（60%）或 35–60 m | **1–4 m**（日志 z = 1.0–3.7） | 起飞 100–150 m（CityNav 设置），全程定高 |
| 地形 | 有起伏（随地形路线一半以上起伏 ≥ 5 m） | 三个校园，平地 | Cambridge / Birmingham 城区；**每张地图只有一个常数地面高度**（`GROUND_LEVEL`） |
| 任务与指令 | 飞到某个物体正上方并 STOP；带起飞方向词 | 短程原子动作："Go past the tree from the left side"、"Orbit the tree counterclockwise"；论文称仿真集多数在 20 m 以内，抽看的真实样本水平位移约 1–36 m、10–45 s | 长航程路线描述，带地标；成功半径 20 m |
| 动作 | dx, dy, dz, dyaw + STOP | 有 RTK 位姿（UTM + 高度 + 姿态），5 Hz，`preprocessed_logs` 是相对首帧的 6 维位姿 → **可以转成 dx/dy/dz/dyaw**；STOP 只能取轨迹末端 | 离散动作：前进 5 m、左右转 30°、STOP；**没有升降** |
| 规模与获取 | 1,000 条路线 | 30,692 条轨迹、178 万帧，**258 GB** parquet，可分片下载 | 约 14 万条；JSON 在 HF `dpairnav/AirNav`；正射图要另从 CityNav/SensatUrban 获取 |
| 许可 | — | 数据卡未写许可（HF 仓库无 README） | 来源数据集（SensatUrban、CityRefer、CityFlight）为 MIT；AirNav 本身待确认 |

## 2. 结论

**按原计划做 D3 不可行。** 原计划是在真实数据上离线评测双视角策略，并支撑地形结论。两个候选都只有单一视角、都没有地形起伏，高度也和我们不在同一量级。UAV-Flow 是 2 m 高度的前视短程动作；AirNav 是 100 m 以上、平地、定高的俯视图，而且图像是正射影像裁出来的。B（双视角）和 T1（地形跟随）在这两个数据集上都无法评测。另查了 UAV-VisLoc（真实俯视、带高度、地形多样），但高度在 400–2000 m，也不匹配。

研究计划的风险表已经预见到这种情况（"不匹配时只声称域迁移或外观鲁棒性，不声称地形结论在真实世界成立"），按那一行执行即可。

## 3. 可行的替代（都只能作描述性证据）

**(a) 真实纹理上的俯视 STOP 探针（推荐，成本低）**
- AirNav/CityNav 的裁剪函数只依赖位姿和正射图，可以在**我们的离地高度**（20–60 m）裁出俯视图，用 CityRefer 的物体（车、建筑、停车场等）构造"目标是否在正下方"的正负样本。
- 用冻结 DINOv2 + 我们的 STOP 头（或 §18 的探针）直接评测，不训练。
- 它对应的是论文里最弱的一环（俯视 STOP），也是我们 benchmark 发现之一：现成模型在真实纹理上同样解决不了。
- 局限：正射影像没有视差，地面是平的；物体类别是城市类，与我们的物体表不同（等同"未见类别"，预计效果差，只作描述）。
- 分辨率（11/05 核实，只读了文件头，未下载）：AirNav 的 `rgbd-new/` 每个街区一对文件，`.png` 是 8 位 RGB 正射图，`.tif` 是同尺寸的单波段 float64 高度图（DSM，nodata = −9999）。GeoTIFF 的 PixelScale 为 **0.1 m/px**（看了 cambridge_block_2：4001×4001；birmingham_block_1：3848×4000；birmingham_block_11：655×1796）。我们的离地高度下，视野边长为 40–120 m，对应原图 400–1200 px，缩到 224 px 后为 18–54 cm/px。**分辨率够用**，全程都是降采样。
- 图像来源：RGB 是由 SensatUrban 的摄影测量点云栅格化成的俯视图，不是原始航拍照片。CityNav 原版用 Potree 渲染点云；SensatUrban 约 30 亿点、7.6 km²，平均约 400 点/m²（估算，相当于点距约 5 cm），所以 10 cm 栅格基本不缺点。但树冠、立面边缘会有摄影测量的拉花和空洞，评测时要抽看。
- 下载量（HF `dpairnav/AirNav`，按 LFS 大小统计）：
  - 只做 (a)：40 张 RGB 正射图 0.77 GB + CityRefer 物体表 0.04 GB ≈ **0.8 GB**。
  - 如果要用 DSM 按屋顶或地面算离地高度（推荐）：再加 40 张高度图 3.72 GB，合计约 **4.5 GB**。
  - 不需要：`gsam/` 3.2 GB（他们的检测缓存）、AirNav 指令 json 0.96 GB、SensatUrban 原始点云（需另外申请）。全仓库共 8.7 GB。
- 目标标注：CityRefer 物体表含 `object_type`（Building、Car、Parking 等）、`position`（中心 xyz）、`dimension`（长宽高）、`contour`（俯视轮廓多边形）和自然语言描述。用 contour 是否覆盖画面中心来判断“目标在正下方”，可以直接构造 STOP 正负样本。注意：建筑常有 30 m 宽，在 20 m 离地高度时会铺满整个视野，所以要按物体尺寸分层报告。

**(b) UAV-Flow 前视外观差异（价值低，不推荐）**：只能比较真实前视和渲染前视的 DINOv2 特征分布，与地形和俯视都无关。

**(c) 真实 DEM + 真实正射影像建图（不是真实图像，属于另一条线）**：用公开 DEM（如 USGS 3DEP）和正射影像在 Blender 里建一张真实地理的地图。它回应的是"地形是人为挑选的"这一风险，而不是 sim-to-real，算力和工时都要另算。

## 4. 决定（11/05）

1. **D3 改为 (a)**。论文里的真实世界证据限定为"真实纹理上的俯视 STOP（描述性）+ sim-to-real 讨论"。
2. **(c) 列为低优先级支线**，主线完成之前不做。
3. 下载等真正开始做 (a) 时再进行，预计 0.8–4.5 GB，本机 3070 就能评测。在那之前要先讨论探针设计：正负样本的定义、物体尺寸分层，以及用 §18 的探针还是 B 的 STOP 头。
