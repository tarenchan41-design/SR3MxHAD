# D4 竞品融合基线：最小复现规格（草案，2026-11-05）

研究计划"11/10 之后：D3/D4 落地"第一项的产出。依据是逐行读过的官方代码，不只看论文：

- AerialVLA（现更名 **AeroVLA**，已中 ECCV 2026）：github.com/XuPeng23/AerialVLA @ `3ba3e24`（2026-10-28），以及 HF `XuPeng23/AerialVLA` 训练 json 的前 30 KB。
- 3DG-VLN（UAV-VLN-FOV，arXiv 2606.20045v1）：github.com/xuefanfu/3DG-VLN @ `0dfd54a`（2026-09-13），以及 HF 模型 config 和数据集 `meta/`。**仓库只有评测代码，没有训练代码**，训练细节取自论文。

复现原则沿用研究计划：同一份冻结 DINOv2-B 特征（或同一 backbone 的新抽取）、同一 prepared 数据、同一 checkpoint 与 STOP 阈值选择规则，**只替换融合模块**。策略头、STOP 头和损失都和 B 相同。

## 1. 对照表

| | **B（本文，HAD）** | **AerialVLA 原版** | **AV（本文复现）** | **3DG-VLN 原版** | **DG（本文复现）** |
|---|---|---|---|---|---|
| 两个视角怎么进模型 | 各自过冻结 DINOv2，各自投影到 512 维，拼成向量 | 两张 224² 上下拼成 224×448，`resize-naive` 压回 224²（每个视角实为 224×112，竖直压缩一半），一张图过 SigLIP+DINOv2 | 同原版：两视角各缩到 224² 后上下拼接，再压回 224²，过冻结 DINOv2-B 一次 | 两张 **1024²** 原图作为两个独立的图像块送进 Qwen2.5-VL-7B（每张约 37×37=1369 个视觉 token） | 每个视角各自过冻结 DINOv2（复用缓存），保留 token：前视 CLS+2×2 = 5 个，俯视 CLS+4×4 = 17 个（或 2×2，按开发图结论） |
| 融合发生在哪 | 拼接后接 MLP（`ConcatFusion`），晚融合、向量级 | **像素级早融合**：冻结视觉骨干内部的注意力就跨了两个视角；再与文本一起在 LLM（LoRA）里联合注意 | 冻结 DINOv2 内部跨视角注意；输出 CLS+2×2（上排约为前视，下排约为俯视）→ 一个共享投影 → 与文本、状态拼接接 MLP | **token 级**：两组图像 token 与文本 token 在 LLM 自注意力里融合 | 视觉 token 与文本词向量一起过 2 层 Transformer，取 [ACT] token |
| 视觉骨干训练 | 冻结 | 冻结（LoRA 只挂 LLM 的 q/k/v/o/gate/up/down_proj；`projector` 全量训练） | 冻结 | LoRA（论文未给 rank；2 epoch，lr 1e-4） | 冻结 |
| 文本 | LSTM 句向量（指令含起飞时的粗方向词） | 指令文本 + **每步方向提示** | 同 B | 指令 + **初始方向提示**，在线更新 | LSTM 词级输出（`word_feats`）作 token |
| 位姿输入 | 有（相对起飞点 5 维） | 无 | 有（同 B） | 无（航点在机体系） | 有（同 B） |
| 动作 | dx, dy, dz, dyaw 回归 | fwd∈[0,5]、down∈[−5,5]、yaw∈[−1.1,1.1]，每维 99 个 bin，以数字文本输出；样本每 5 帧取一张 | 同 B | 5 个机体系航点 [dx,dy,dz]，无 yaw；由 `moveOnPathAsync`（ForwardOnly，1 m/s）执行，航向跟随路径切线 | 同 B |
| STOP | 学习的 STOP 头 | 学习：最后两步标签加 `LAND`；推理时输出 LAND，或三个 bin 恰为 (0,49,49)，都判停 | 同 B | **不学**：GroundingDINO（框阈值 0.6）在两视角检测，框中心深度 < 8 m 即停 | 同 B |
| 俯视角的特殊用法 | 无 | 无 | 无 | 俯视单框检测 → 方向改写为 "It's down below you"，且俯视优先 | 无 |

**一句话**：AV 测“像素级早融合 + 共享冻结编码器”，DG 测“token 级联合注意力”，B 是“分视角编码 + 晚融合拼接”。三者的输入特征、数据、头、选择规则都相同，差别只在融合方式。

## 2. 刻意不复现的部分（论文里要说明）

1. **方向提示（两家都有，都来自真值目标位置）**
   - AerialVLA：`get_semantic_direction(curr_state, target_pos)` 每一步用真值目标位置算机体系方位角，分 7 个扇区写进 prompt，评测时也一样（`aerovla_wrapper_ui.py:46–76, 92`）。论文摘要称其“derived solely from onboard sensors”，与代码不符。
   - 3DG-VLN：初始方向（straight ahead / front-left / front-right / down below you）由初始位姿和目标位置算出，论文正文明说。测试集（489 条，`meta/instructions.json`）分布为 straight ahead 333、down below you 65、front-right 53、front-left 38。
   - 我们的指令已带**起飞时**的粗方向词（§13 规则 7：|方位角| < 12° 为 "ahead of you"，否则 "ahead on your left/right"；起飞方位角在 ±35° 内，且直飞会错过目标），与 3DG-VLN 的初始方向同级，三个变体都用。不复现的是**逐步更新**的方向（AerialVLA 每步一次，3DG-VLN 靠检测器更新）：在闭环里每一步都要知道目标真值位置；目标到正下方时（3DG-VLN 的 "down below you"）等于直接告诉模型该停了。
2. **骨干与规模**：OpenVLA-7B / Qwen2.5-VL-7B 加 LoRA 超出低算力原则。我们复现的是“融合拓扑”，不是“大模型”。
3. **动作表示**：99 bin 文本动作、5 航点输出、ForwardOnly 路径跟随都属于策略头或执行器，不属于融合，统一用我们的动作合同。
4. **检测器 STOP 与在线方向更新**（3DG-VLN）：属于外挂模块，不属于融合。见第 5 节的可选项。
5. **高分辨率**：3DG-VLN 的 LR 消融（256² vs 1024²）在 Test 上掉 15.8 个点 SR，是它最大的单项收益。我们的渲染是 512²，统一在 224² 上抽特征。DG 默认不做高分辨率，见第 5 节的 DG-HR。

## 3. 实现规格

在 `had/model.py` 里加两个变体，**不改 `had/vendor/`**。

### AV
- `VARIANTS["AV"] = ("mosaic",)`。特征：`mosaic` = 冻结 DINOv2-B 作用于拼图后的 CLS+2×2（3840 维）。
- 拼图：前视、俯视各 `Resize((224,224), bicubic, antialias)` → 上下拼成 224×448（前视在上）→ 现有 `image_transform()` 的 `Resize((224,224))` 直接压缩，不裁剪。这与 OpenVLA 的 `resize-naive` 一致（已核对 HF `preprocessor_config.json`）。
- 投影：一个 `ViewProjection(512, 3840)`。融合：新类 `MosaicFusion`，结构同 `ConcatFusion`，但 `in_dim = vis + text + height`。之后的 `state_fusion`、`policy` 不变。
- 需要**新抽一次特征**（62,177 个位姿 × 1 次 DINOv2 前向，与抽一个视角同量级）。见第 6 节问题 1。

### DG
- `VARIANTS["DG"] = ("front", "down")`。复用现有缓存：前视 2×2，俯视 4×4（若开发图选 2×2，就用 2×2）。缓存里的 3840 / 13056 维向量按 768 维 reshape 成 5 / 17 个 token，**不需要新抽**。
- 每个 token 过 `Linear(768→256)+LN`；文本 `word_feats` 过 `Linear(text_dim→256)+LN`，带 padding mask。
- 加可学习的类型嵌入（front / down / text）和网格位置嵌入（视觉 token 按格子编号）；在最前面放一个可学习的 [ACT] token。
- 2 层 pre-norm `TransformerEncoder`（d=256、4 头、FFN 1024、dropout 0.2）→ 取 [ACT] → `Linear(256→512)` 得到 `fused` → 现有 `state_fusion` 和 `policy`。
- 结构类比：Qwen2.5-VL 中两块图像 token 与文本 token 的联合自注意力。

### 两者都要满足
- A/C 式视图屏蔽不适用（AV 只有一个输入；DG 可以另做“mask 俯视 token”的诊断，但不进主表）。
- 单元测试：形状、`set(features) == views` 检查、`head_state` 往返、小配置一步反传；打印可训练参数量写进结果表（B 约为 2×ViewProjection + ConcatFusion，DG 用 d=256 时量级相近，实测后再报）。
- 训练：与 B 相同的数据、种子 42/43/44、选择规则和 STOP 校准。只在训练图的 train/val_seen 上训练和选模，开发图、测试图的使用次数计入 §17.9 / §17.10 的预算，不另开预算。

## 4. 对论文的直接含义（事实，可写进相关工作）

- **两家都没有做“只用前视 vs 双视角”的消融**。AerialVLA 只比了 5 视角（更差：Unseen Map SR 37.58 → 21.71）；3DG-VLN 只比了分辨率和方向提示。我们的 A/B/C 对比和“按地形复杂度看俯视收益”没有被做过。
- 两家的俯视角主要服务终点：AerialVLA 的 LAND，3DG-VLN 的 "down below you" 改写和检测器停止。这与我们“俯视 STOP 难、现成图文模型解决不了”的发现一致。值得注意，3DG-VLN 直接把俯视 STOP 外包给了 GroundingDINO + 深度。
- 两家都依赖由真值目标位置算出的方向提示。表述时建议写成“uses a coarse target-direction prompt computed from the target position”，只陈述事实。注意我们的起飞方向词也由目标位置算出（相当于下达指令的人知道目标在哪边），差别只在**是否逐步更新**，不能写成“我们不用方向提示”。
- 评测口径不同，不能直接比数：
  - AerialVLA 用 SR20；距离连续 10 步增大就判为碰撞并终止；误停在 20 m 外不结束回合，但之后不能再算成功。
  - 3DG-VLN 用 SR10，STOP 来自检测器。
- 两家都没有离地高度或地形指标（11/05 已核实）。

## 5. 可选项（默认不做，需要你决定）

- **DG-HR**：在 448² 上重抽 DINOv2（每个视角 32×32 patch，再池化到 4×4 或 8×8），检验“高分辨率”这一 3DG-VLN 主卖点在我们任务上是否成立。前向成本约为 224² 的 4 倍，在 5090 上跑。
- **检测器 STOP 基线**（3DG-VLN 式）：GroundingDINO 在俯视和前视上检测目标，框内距离小于阈值即停。可以放进“现成模型解决不了俯视 STOP”那组 benchmark 发现里（目前只测了 SigLIP、RemoteCLIP、Qwen3-VL 零样本）。距离用 K 中已知的地形网格射线求交或深度渲染，需另行确认。

## 6. 待你决定

1. **AV 的拼图特征是否并进阶段 2 的抽特征**（`experiments/cache.py` 已经在同一遍里出 4×4 俯视，再加一个输出最省事）。代码改动要等渲染结束后再提交，然后才能开始抽特征；否则以后要单独再抽一遍。
2. 俯视 token 粒度跟随开发图的 2×2 / 4×4 结论，还是 DG 固定用 4×4。
3. 第 5 节两个可选项做不做。
