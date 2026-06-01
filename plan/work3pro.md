# work3 Pro 方向更新：LiDAR-Privileged Spatial Token Pretraining for Camera-Only Driving VLMs

> 更新时间：2026-05-31
>
> 本文档将 work3 Pro 从“多任务几何蒸馏框架”收敛为一个更适合 CVPR 2027 的研究方向：
>
> **用训练阶段可用的 LiDAR ego-centric 几何关系，预训练一个 language-accessible 的 camera-only spatial token 生成器，并用真实 spatial QA / spatial probing / robustness 实验证明这些 token 对 frozen driving VLM 有用。**

---

## 1. 新的一句话定位

**LiDAR-privileged relation pretraining for language-accessible camera tokens in autonomous driving.**

中文：

**利用训练阶段的 LiDAR 特权几何关系监督，训练一个纯视觉空间 token 生成器，让 camera-only driving VLM 在推理时仅凭摄像头也能获得更可靠的距离、方位、遮挡、free space、跨视角和鲁棒空间推理能力。**

更严谨的一句话：

> **GeoToken uses LiDAR-derived ego-centric spatial relations to pretrain camera tokens, making them encode metric and topological cues useful for ego-centric driving reasoning.**

更像论文标题的版本：

> **Privileged Spatial Tokens: LiDAR-Supervised Camera-Only Representation Learning for Driving VLMs**

或者：

> **GeoToken: LiDAR-Privileged Spatial Token Pretraining for Camera-Only Autonomous Driving VLMs**

写作时注意：不要说 token 本身“就是 ego 坐标系里的点”。更准确的说法是：camera tokens 被 LiDAR-derived ego-centric relations 训练后，**携带 ego-centric spatial structure**，因此更容易被 VLM 用于驾驶空间推理。

---

## 2. 为什么要更新方向

原 work3 关注跨模态视觉位置识别。新版 work3 Pro 不再沿着 VPR、descriptor distillation 或 localization 方向继续扩展，而是完全转向 driving VLM 的三维空间感知问题。

因此新版方向必须收敛到：

> 不做地点识别，不做 descriptor 蒸馏，不做泛泛的多头几何辅助学习，而是做 **面向 driving VLM 空间推理的 LiDAR 特权 token 预训练**。

---

## 3. 2026-2027 CVPR 趋势判断

截至 2026-05，自动驾驶 + VLM 的方向已经从“VLM 能不能参与驾驶”转向下面几个更硬的问题。

### 3.1 从 VLM 到 VLA / closed-loop driving

代表方向：

- SimLingo：vision-only VLM + language-action alignment + closed-loop driving。
- DriveVLM / OpenDriveVLA / NaviDriveVLM 类方向：VLM 不只回答问题，还要参与 planning / action。
- Bench2Drive / CARLA closed-loop 逐渐成为 VLM driving 说服力来源。

审稿倾向：

> 单纯说“接 VLM”不够，必须证明 representation 对 decision、action、QA 或 planning-relevant proxy 有帮助。

work3 Pro 应对：

> 不承诺第一篇做完整 closed-loop，而是做 **planning-relevant spatial reasoning**：距离、方位、可通行区域、遮挡、路口结构、cross-view object relation。

### 3.2 从语义理解到空间 grounding

代表方向：

- SpaceDrive：明确提出给 VLM-based driving 注入 spatial awareness。
- SpatialVLM / VLM-3R 类方向：让 VLM 具备 3D / depth / reconstruction / metric reasoning。
- DriveSpatial / STSBench 类 benchmark：强调 cross-view、spatio-temporal、scene graph、temporal correspondence。

审稿倾向：

> 2027 年只做“场景描述更好”价值会变弱，必须证明模型真的理解 3D 空间关系。

work3 Pro 应对：

> 主实验直接聚焦 **spatial token 是否提升 VLM 的 metric / relational / temporal QA**。

### 3.3 从 perception benchmark 到 pretraining benchmark

代表方向：

- VisionPAD 等 vision-centric pretraining：预训练不再只服务 detection，还服务 occupancy、map、motion、BEV。
- 3D occupancy、BEV map、world model、future prediction 成为 driving foundation model 的中间表示。

审稿倾向：

> 一个 representation learning paper 必须证明 frozen representation 能迁移到多个下游任务。

work3 Pro 应对：

> 采用 frozen token probing：depth、BEV occupancy / map、pairwise relation、occlusion order、driving spatial QA。

### 3.4 从 clean benchmark 到 robustness benchmark

代表方向：

- RoboDriveVLM 类工作强调 VLM driving 在传感器腐蚀、环境变化、prompt corruption 下的风险。
- camera-only driving 对 limited FOV、camera drop、night、rain、motion blur 极敏感。

审稿倾向：

> robust evaluation 会从加分项变成必要项。

work3 Pro 应对：

> sensor failure 不再只是 augmentation，而是论文的第三主实验：clean vs corrupt 下 token 和 VLM 表现的下降幅度。

### 3.5 从大模型堆算力到可部署 token

代表方向：

- 小 VLM、efficient VLM、distillation、on-vehicle deployment 是 CVPR workshop 和应用论文的重要趋势。

审稿倾向：

> 比端到端训练巨大 VLM 更可接受的是：提出一个可插拔、低成本、能改善现有 VLM 的视觉 token 模块。

work3 Pro 应对：

> 第一篇不训练大 VLM，而是冻结 Qwen-VL / InternVL / LLaVA-Video 类模型，只训练轻量 projector 或 token adapter。

---

## 4. 新版核心问题

### 4.1 主问题

**Can LiDAR-privileged training produce camera-only visual tokens that measurably improve spatial reasoning and robustness of driving VLMs?**

中文：

**训练阶段的 LiDAR 几何监督，能否产生一种纯视觉 token，使 driving VLM 在空间推理和退化场景下明显更可靠？**

### 4.2 子问题

1. 如何把 LiDAR 的 metric 3D geometry 转成 camera token 可学习的监督？
2. 如何让 token 不只是学到普通 depth / BEV 辅助信息，而是真正提升 VLM spatial QA？
3. 如何避免方法变成普通 depth / BEV auxiliary loss？
4. 如何在不训练大 VLM 的情况下，证明 token 对 driving VLM 有价值？
5. 如何证明 camera drop、night、rain、limited FOV 下 token 更稳？

---

## 5. 推荐论文主线

新版 work3 Pro 不再写成“多任务 token 输出框架”，而写成：

```text
Training:
  Multi-view camera video
        ↓
  Camera spatial tokenizer
        ↓
  camera-only spatial tokens

  LiDAR sequence + ego pose
        ↓
  LiDAR geometric graph / BEV / range teacher
        ↓
  privileged spatial relations

  Distillation:
    3D metric relation graph → camera token relation graph
    BEV occupancy prior      → camera spatial tokens
    temporal ego-motion      → camera temporal tokens
    full-camera tokens       → degraded-camera tokens

Inference:
  multi-view camera only
        ↓
  spatial tokens
        ↓
  VLM adapter / spatial QA / relation probing / BEV probing
```

关键表达：

> LiDAR is not a deployment sensor; it is a training-time spatial examiner.

---

## 6. 核心创新应收敛为一个机制

### 6.1 建议命名

建议把核心方法命名为：

**Privileged Spatial Relation Distillation, PSRD**

中文：

**特权空间关系蒸馏**

它比“depth loss + BEV loss + token loss”更像一个可投稿的核心贡献。

### 6.2 PSRD 的核心思想

不要只让 camera token 回归 LiDAR feature，而是让 camera token 学 LiDAR 中的 **空间关系结构**。

LiDAR teacher 产生四类关系：

1. **metric distance relation**
   - object / patch / BEV cell 之间的真实距离远近关系。

2. **directional relation**
   - front / rear / left / right / relative bearing。

3. **occupancy-topology relation**
   - free space、occupied space、drivable topology、junction / lane structure。

4. **occlusion / visibility order relation**
   - 哪个 object / region 在视线方向上更靠近相机或 ego，哪个目标被前景遮挡。

Camera student 不直接复制点云，而是学习：

> 哪些 token 在 3D 空间中接近，哪些远离，哪些有遮挡，哪些属于同一可通行区域。

关键表述：

> GeoToken does not make image tokens literal coordinates. It uses LiDAR-derived ego-centric relations to make camera tokens encode metric, directional, topological, and occlusion cues that a driving VLM can query through language.

### 6.3 PSRD 的一个可实现版本

对每个 clip 的中心帧：

1. 用 LiDAR + ego pose 构建 BEV grid。
2. 主图节点统一为 ego-centric BEV anchor token，不混用 image patch / object region / BEV cell。
3. 将 BEV anchor 投影到 multi-view camera patch，用 calibration-aware cross-attention 从图像特征读出 anchor token。
4. object box 和 image patch 只用于标签、可见性、语言短语和评测，不作为 PSRD 主图节点。
5. 为 BEV anchor 建立 3D relation graph：

```text
G_lidar = (V, E)
V_i = BEV anchor at ego-frame cell center (x_i, y_i)
E_ij = [
  distance_bin(i, j),
  relative_angle(i, j),
  same_occupancy_region(i, j),
  occlusion_order(i, j),
  reliability_mask(i, j),
  occlusion_reliability_mask(i, j)
]
```

这样 reviewer 问“node 到底是什么”时，回答是：

> PSRD uses only one node type: fixed ego-centric BEV anchors. Object regions and image patches are projected evidence and pseudo-label sources, not graph nodes.

推荐默认范围：

- ego-frame local range：`x in [-40m, 40m]`, `y in [-40m, 40m]`。
- candidate grid size：`2m x 2m`，得到 `40 x 40 = 1600` candidate anchors。
- 训练时从 1600 个 candidate anchors 中选择 `L=400` active anchors。
- active anchor 优先级：3D box 覆盖 anchor > occupied LiDAR evidence > drivable/free-space boundary > camera-frustum visible anchor > uniform free-space sample。
- 如果不足 400，用 visible free-space anchors padding；如果超过 400，按优先级和空间均匀性下采样。
- 推理时不能使用 LiDAR / GT box filtering。推理有两种选择：直接用 chunked attention 跑完整 1600 anchors，或用 tokenizer 预测的 camera-only priors 选 400 个 anchors，例如 camera-frustum visibility、learned occupancy scores、learned objectness scores。
- object-level 表征：pool 3D box 覆盖到的 anchors。
- free-space 表征：pool 同一 free-space component 内的 anchors。

6. Camera student 输出 BEV anchor token relation：

```text
R_cam = softmax(Q_cam K_cam^T / sqrt(C))
```

7. 蒸馏 LiDAR relation：

```text
L_psrd = KL(R_lidar || R_cam)
       + SmoothL1(D_cam_pair, D_lidar_pair)
       + CE(A_cam_pair, A_lidar_pair)
       + CE(O_cam_pair, O_lidar_pair)
       + CE(Q_cam_pair, Q_lidar_pair)
```

其中：

- `R_lidar`：由 LiDAR 3D 距离 / BEV 拓扑得到的 soft relation。
- `D_lidar_pair`：pairwise metric distance bin 或相对深度等级。
- `A_lidar_pair`：方向关系类别，例如 front-left / front-right / rear。
- `O_lidar_pair`：occupancy / topology 类别，例如 same free-space region、same occupied component、different component。
- `Q_lidar_pair`：visibility / occlusion order 类别，例如 i-before-j、j-before-i、same-depth；无法可靠判断的 pair 用 `m_occ_ij=0` 排除。

这比单纯 `MSE(BEV_camera, BEV_lidar)` 更像新的方法。

### 6.4 PSRD 标签定义

默认离散化：

```text
distance bins:
  [0,2), [2,5), [5,10), [10,20), [20,40), >=40 meters

direction bins:
  same-cell,
  front, front-left, left, rear-left,
  rear, rear-right, right, front-right

topology classes:
  same-free-space,
  same-occupied-component,
  free-to-occupied,
  occupied-to-free,
  different-free-components,
  unknown

occlusion classes:
  i-before-j,
  j-before-i,
  same-depth
```

`occlusion_order` 只在两个 anchors 投影到同一 camera ray 邻域或相邻 ray 邻域，并且深度可分时计算。否则 `m_occ_ij = 0`，不参与 `L_occ`。

### 6.5 Reliability mask `m_ij`

`m_ij = 1` 的条件：

- 两个 anchors 都在 local driving range 内。
- 两个 anchors 至少被一个 camera 覆盖，且投影落在有效图像区域。
- anchor 有足够 LiDAR 支持，或能由 ray casting 得到可靠 free-space / occupied 证据。
- pair 的 relation label 不跨越明显不同步的动态 object association。

`m_ij = 0` 的情况：

- LiDAR 点太稀疏。
- calibration projection 落出图像。
- 多 camera 投影冲突且无法确定可见性。
- 远距离小物体导致 box / LiDAR association 不稳定。

`m_occ_ij = 1` 需要更严格：

- 两个 anchors 投影到同一 camera 或相邻 camera 的 ray neighborhood。
- 两个 anchors 的深度差超过阈值，例如 `0.5m`。
- 没有明显 calibration conflict 或 multi-view visibility conflict。
- 如果不满足，则只用于 distance / direction / topology，不用于 occlusion loss。

### 6.6 Pair sampling，避免 `O(L^2)`

PSRD 不在所有 `L^2` pairs 上训练。对每个 anchor `i` 采样最多 `P=32` 条边：

- `P_near`：metric 最近邻。
- `P_topo`：同一 free-space 或 occupied component 内的 anchors。
- `P_occ`：有有效遮挡顺序的 anchors。
- `P_hard`：不同 topology component 但图像外观相似的 hard negatives。
- `P_rand`：随机远距离 valid anchors。

默认设置：

```text
active anchors L = 400
edges per anchor P = 32
edges per frame <= 12.8k
```

这样训练复杂度是 `O(LP)`，不是 `O(L^2)`。dense pair 只用于小规模 probing 或可视化。

### 6.7 `rho(o_ij)` 和 loss weights

`R_lidar` 中 topology affinity 的默认设计：

```text
rho(o_ij) =
  1.5  same-free-space
  1.3  same-occupied-component
  0.8  free-to-occupied / occupied-to-free
  0.5  different-free-components
  0.0  unknown
```

距离衰减：

```text
alpha = 1 / 20
```

含义是 20m 处 affinity 约为 `exp(-1)`。

默认 PSRD 权重：

```text
λ_rel  = 1.0
λ_dist = 1.0
λ_dir  = 0.5
λ_topo = 0.5
λ_occ  = 0.5
```

如果 topology / occlusion 类别长尾明显，`L_topo` 和 `L_occ` 使用训练集上的 inverse-frequency class weights。

---

## 7. 模型设计

### 7.1 Camera Spatial Tokenizer

建议结构：

```text
DINOv2 / SigLIP / CLIP-ViT frame encoder
        ↓
View embedding + camera-id embedding
        ↓
Multi-view token fusion
        ↓
Temporal encoder
        ↓
Spatial token adapter
        ↓
Outputs:
  - BEV anchor tokens
  - spatial slots
  - adapter tokens
  - temporal tokens
```

术语统一：

- `image patch tokens`：图像 backbone 输出。
- `BEV anchor tokens`：PSRD 主监督对象。
- `spatial slots`：LGSS 输出，用于 object / region / relation 语言访问。
- `adapter tokens`：投影后输入 frozen VLM 的 tokens。

第一篇建议：

- Backbone：DINOv2 ViT-S/14 或 ViT-B/14。
- Fusion：view-aware token attention。
- Temporal：lightweight transformer，T=2/3/5 起步。
- 不保留 VPR descriptor head；GeoToken 的输出服务于 VLM spatial reasoning。
- BEV anchor tokens：保持 384/768-D。
- VLM adapter：只做轻量 projector，不训练大 VLM。

### 7.2 LiDAR Geometric Teacher

不建议第一篇上来就换很复杂的 3D backbone。

分两级：

**MVP teacher**

- range image encoder 或现有 `T_net.py`。
- 输出 range / BEV geometry feature。
- 用 LiDAR 投影生成 depth / pairwise distance / visibility relation。

**strong teacher**

- BEV encoder / RangeViT / SphereFormer / occupancy encoder。
- 输出 BEV occupancy feature 和 3D relation graph。

论文可以先用 MVP teacher 做主结果，strong teacher 做 upper bound 或 appendix。

### 7.3 VLM Adapter

第一篇不要承诺端到端训练大 VLM。

推荐：

```text
spatial tokens → Q-Former / MLP projector → frozen VLM hidden space
```

训练方式：

- 冻结 camera tokenizer。
- 冻结 VLM。
- 只训练 projector / small adapter。
- 在 DriveLM-nuScenes / 自构造 spatial QA / DriveSpatial-style QA 上评测。

这样 claim 更稳：

> Our tokens are VLM-usable, not merely VLM-compatible.

### 7.4 LGSS：语言可访问的空间 slot

PSRD 只能证明 camera tokens 学到了空间结构，但不能自动保证 frozen VLM 能用自然语言访问这些结构。因此建议保留 **Language-Grounded Spatial Slots, LGSS**，但把它写成接口机制，而不是第二个同等贡献。

推荐结构：

```text
spatial tokens Z
        ↓
learnable spatial queries
        ↓
cross-attention slot reader
        ↓
spatial slots S
        ↓
weak spatial phrases / QA adapter / frozen VLM
```

slot 不再笼统写成“一组 slots”。建议拆成三类：

```text
object slots:
  读 3D box 覆盖到的 anchors，用于 vehicle / pedestrian / cyclist 等目标。

region slots:
  读 free-space / occupied / drivable / junction-like area。

relation slots:
  读 object-object、object-region、region-region 的关系，
  用于 closer/farther、left/right、occlusion、cross-view correspondence。
```

这样回答 reviewer：

> One slot is not forced to represent everything. Object slots, region slots, and relation slots have separate query embeddings and supervision.

弱语言短语来自 LiDAR / 3D box / BEV relation labels，例如：

- `a nearby vehicle in front`
- `free space on the front-left side`
- `the farther vehicle behind the leading car`
- `a pedestrian occluded by the front vehicle`

### 7.5 LGSS slot assignment

text encoder 不能随便写成 frozen text encoder。选择原则：

- LGSS 使用与目标 VLM interface 对齐的 frozen text representation。
- CLIP-aligned VLM 使用 paired CLIP text encoder。
- LLaVA / InternVL / Qwen-VL-style 模型使用其 tokenizer 和 language embedding stack，再接 lightweight projection 到 adapter space。
- 如果目标 VLM 不是 CLIP-aligned，CLIP text encoder 只作为 controlled ablation。

slot assignment 使用 Hungarian matching，而不是默认 slot 自动学会：

```text
cost(s_k, y_j) =
  λ_iou  (1 - IoU(A_k, A_j))
+ λ_cls  CE(c_hat_k, c_j)
+ λ_txt  (1 - sim(s_k, t_j))
```

其中：

- `A_k`：slot 到 BEV anchors 的 attention map。
- `A_j`：pseudo target mask，来自 3D box 或 BEV region。
- `c_j`：object / region class。
- `t_j`：对应 spatial phrase 的 frozen text embedding。

phrase 多义时用 multi-positive contrastive：

```text
L_slot = -log sum_{p in P(i)} exp(sim(s_i, t_p) / τ)
              / sum_j exp(sim(s_i, t_j) / τ)
```

例子：

- `a vehicle in front` 可能对应多个前方车辆，全部作为 positives。
- `free space on the left` 可能覆盖多个 connected free-space components，保留所有满足条件的 region positives。

slot 去重：

```text
L_div = || A A^T - I ||_F
```

其中 `A` 是 normalized slot-to-anchor attention matrix，避免多个 slots 全部读同一个 object。

时序一致性：

```text
L_slot_temp = || Warp(A_t, T_t→t+1) - A_t+1 ||_1
```

用 ego-motion compensation 后的 attention map 对齐，减少同一目标在相邻帧 slot identity 跳变。

LGSS 总损失：

```text
L_lgss =
  λ_slot      L_slot
+ λ_qa        L_qa
+ λ_div       L_div
+ λ_slot_temp L_slot_temp
```

默认权重：

```text
λ_slot      = 1.0
λ_qa        = 1.0
λ_div       = 0.05
λ_slot_temp = 0.1
```

必须防止 LGSS 被质疑为模板记忆：

- 训练短语和测试问句做 paraphrase split。
- 训练对象组合和测试对象组合做 hard split。
- 同一个空间关系用多种同义表达。
- 报告 template split 与 paraphrase split 两套结果。
- 做 no-LGSS、no-diversity、no-temporal-slot-consistency ablation。

更稳的 claim：

> LGSS makes the LiDAR-privileged spatial tokens more accessible to language, but the core geometric learning still comes from PSRD.

---

## 8. 损失函数重构

旧版总 loss 太散。新版建议拆成主损失和辅助损失。

### 8.1 主损失：PSRD

```text
L_psrd =
  λ_rel  L_relation
+ λ_dist L_metric_distance
+ λ_dir  L_direction
+ λ_topo L_occupancy_topology
+ λ_occ  L_occlusion_order
```

这是论文的核心。

注意：如果论文评测 occlusion，就必须保留 `L_occlusion_order`。否则要删除遮挡 claim，避免“训练目标没有遮挡监督，但实验主张遮挡提升”的断裂。

### 8.2 辅助损失

```text
L_aux =
+ λ_temp  L_temporal_ego_motion
+ λ_cons  L_sensor_consistency
```

辅助损失只服务主线，不要每个都包装成 contribution。depth / BEV 可以作为 probing head 或 ablation，不建议作为默认 pretraining 主损失，否则主线会偏回普通几何辅助学习。

### 8.3 总损失

```text
L_total = L_psrd + L_aux
```

论文写作中必须强调：

> Depth / BEV auxiliary objectives are not the novelty; privileged spatial relation distillation is.

---

## 9. 实验必须重新排序

CVPR 2027 版本的实验顺序应该是：

1. VLM spatial reasoning
2. Frozen spatial probing
3. Robustness under degradation

VPR / localization 不进入主实验，也不作为附录实验。本文与 VPR 任务脱钩。

### 9.1 主实验 1：Driving VLM Spatial QA

目标：

> 证明 spatial tokens 真的改善 VLM 的空间推理。

任务类型：

- 距离判断：哪辆车更近？
- 方位判断：行人位于 ego 的哪个方向？
- 可通行区域：左侧是否有可通行空间？
- 遮挡判断：前车是否遮挡了远处车辆？
- cross-view 关系：左前方目标是否同时出现在 front-left 和 front camera？
- temporal 判断：目标是在靠近还是远离？

数据来源：

- DriveLM-nuScenes。
- 自构造 nuScenes spatial QA。
- DriveSpatial / STSBench-style spatio-temporal QA，如果可用则加入。

防泄漏设置：

- train QA templates 与 test QA templates 分离。
- report normal template split 和 hard paraphrase split。
- hard split 中同一关系不能只换 object 名称，必须换语言表达方式，例如训练 `a nearby vehicle in front`，测试 `the car closest ahead of the ego vehicle`。

对比：

- frozen VLM + original visual tokens。
- frozen VLM + DINOv2 tokens。
- frozen VLM + depth-prediction pretraining tokens。
- frozen VLM + BEV auxiliary pretraining tokens。
- frozen VLM + work3 Pro PSRD tokens。

指标：

- accuracy。
- spatial subset accuracy。
- distance / direction / topology / occlusion / cross-view / temporal 分项。
- corruption 下 accuracy drop。

### 9.2 主实验 2：Frozen Spatial Probing

目标：

> 不靠 VLM 微调，证明 representation 内部有空间信息。

任务：

1. Depth probing
   - frozen tokens + small head。
   - AbsRel / RMSE / δ。

2. BEV occupancy / map probing
   - frozen tokens + BEV head。
   - IoU / mIoU。

3. Object relation probing
   - pairwise relation classifier。
   - distance bin / direction bin / occlusion order accuracy。

### 9.3 主实验 3：Robustness

目标：

> 证明 LiDAR-privileged spatial token 比普通 visual token 更抗退化。

设置：

- clean。
- single camera drop。
- multiple camera drop。
- front-only。
- limited FOV。
- random occlusion。
- night。
- rain。
- motion blur。

指标：

```text
Robustness Ratio = Performance_corrupt / Performance_clean
Average Robust Accuracy = mean(Acc over corruptions)
```

需要同时报告：

- VLM spatial QA robustness。
- relation / depth / BEV probing robustness。

### 9.4 不做 VPR / Localization

本文不做 VPR、place recognition、localization retention 或 descriptor-only distillation。实验闭环只围绕 camera-only VLM spatial reasoning：

- driving spatial QA；
- relation / depth / BEV probing；
- occlusion / cross-view / temporal reasoning；
- camera degradation robustness。

---

## 10. 数据集策略

### 10.1 第一优先级：nuScenes

原因：

- 6-camera + LiDAR + ego pose。
- 有 DriveLM-nuScenes 生态。
- 有 night / rain / location split。
- 适合 spatial QA、relation probing、BEV probing、sensor failure。

使用方式：

- clip 输入：`B × T × N × 3 × H × W`。
- LiDAR 只在训练阶段使用。
- ego pose 用于 temporal / relation label。
- 中心帧构造 spatial QA 和 relation label。

### 10.2 第二优先级：KITTI-360

原因：

- 长序列。
- 适合 temporal / geometry probing。
- 可做 cross-dataset generalization。

注意：

- 不要强行套 6-camera 设定。
- 用作泛化验证，而不是第一主战场。

### 10.3 可选：Bench2Drive / CARLA

如果资源允许，用于 planning-relevant proxy：

- closed-loop 不作为第一篇必要项。
- 可以用 QA / trajectory prediction / action classification 作为补充。

---

## 11. Baseline 必须更强

为了应对 CVPR 2027 审稿，baseline 不能只和原 work3 比。

### 11.1 Token baseline

- CLIP / SigLIP tokens。
- DINOv2 tokens。
- DINOv2 + temporal encoder。
- DINOv2 + depth auxiliary。
- DINOv2 + BEV auxiliary。
- DINOv2 + sensor dropout。
- work3 Pro PSRD。

### 11.2 VLM baseline

- frozen Qwen-VL / InternVL / LLaVA-Video 原始视觉输入。
- frozen VLM + DINOv2 adapter。
- frozen VLM + BEV/depth-pretrained adapter。
- frozen VLM + PSRD adapter。

### 11.3 Driving baseline

- SimLingo / DriveVLM / SpaceDrive 等作为 conceptual related work。
- 如果无法复现，不作为直接 baseline。
- 直接实验用可控的 frozen VLM + same adapter protocol。

---

## 12. Ablation 设计

必须证明提升来自 PSRD，而不是更大 backbone 或更多数据。

必做 ablation：

1. no LiDAR privileged supervision。
2. depth-only distillation。
3. BEV-only distillation。
4. token relation without metric distance。
5. token relation without direction。
6. token relation without occupancy topology。
7. token relation without occlusion order。
9. no LGSS。
10. LGSS without paraphrase split。
11. LGSS with CLIP text encoder vs target-VLM text space。
12. LGSS with mismatched text encoder。
13. LGSS with shuffled phrases。
14. no temporal relation。
15. no sensor consistency。
16. different T：1 / 3 / 5。
17. different camera views：6-view / 3-view / front-only。

关键表：

```text
Method              VLM Spatial QA  Relation Probe  Depth Probe  BEV Probe  Robust QA
DINOv2              ...
+ Depth             ...
+ BEV               ...
+ PSRD no topology   ...
+ PSRD full          ...
```

---

## 13. 论文贡献点新版

建议只写 3 个 contribution。

1. **Problem formulation**
   - 提出 LiDAR-privileged relation pretraining，用于训练 language-accessible camera-only spatial tokens，提升 driving VLM 的空间推理和鲁棒性。

2. **Privileged Spatial Relation Distillation**
   - 将 LiDAR 3D geometry 转成 metric / directional / occupancy-topology / occlusion-order relation graph，并蒸馏到 camera video tokens。

3. **VLM-centered evaluation and language access**
   - 通过 LGSS 和 frozen VLM adapter 验证 spatial tokens 能被语言访问，并在 nuScenes / KITTI-360 上通过 VLM spatial QA、frozen spatial probing 和 sensor degradation 系统验证。

不要把 VLM projector、depth head、BEV head 写成 contribution。它们是实验接口。

---

## 14. 和原 work3 的关系

新 work3 Pro 与原 work3 在任务定义上脱钩：

- 原 work3 是跨模态视觉位置识别。
- 新 work3 Pro 是 driving VLM spatial token pretraining。
- 新论文不使用 VPR、place recognition、localization 或 descriptor distillation 作为实验任务。
- 原代码只作为数据读取和工程参考，不作为论文叙事或实验基线。

---

## 15. 最小可投稿版本

如果目标是 CVPR 2027，MVP 不应该只是 shape test，而应该是能支撑论文 claim 的最小闭环。

### 15.1 Method MVP

必须有：

1. `NuScenesClipDataset`
2. DINOv2 / SigLIP multi-view camera tokenizer
3. temporal encoder
4. LiDAR projection to camera patch / BEV cell
5. PSRD relation label construction
6. PSRD loss
7. frozen VLM adapter

可以暂缓：

- full BEV teacher。
- large VLM full fine-tuning。
- closed-loop CARLA。
- Boreas。
- very strong LiDAR 3D backbone。

### 15.2 Experiment MVP

必须有：

1. nuScenes VLM spatial QA
2. depth probing
3. BEV or occupancy probing
4. camera drop / night / rain robustness
5. PSRD ablation

没有第 1 项，就不要把标题写成 driving VLM。

---

## 16. 代码路线新版

### Phase 1：clip dataset + video tokenizer

产出：

- `dataset/NuScenesClipDataset.py`
- `modules/dinov2_backbone.py`
- `modules/multiview_fusion.py`
- `modules/bev_anchor_queries.py`
- `modules/calibration_cross_attention.py`
- `modules/temporal_encoder.py`
- `modules/S_net_video.py`
- `tests/test_video_student_shapes.py`

目标：

```python
output = model(camera_clip)
assert output["spatial_tokens"].shape[:2] == (B, L_active)
assert output["temporal_tokens"].ndim == 3
```

### Phase 2：LiDAR relation label builder

产出：

- `dataset/lidar_projection.py`
- `dataset/bev_anchor_grid.py`
- `dataset/spatial_relation_builder.py`
- `tests/test_spatial_relation_builder.py`

目标：

```python
relations = build_spatial_relations(camera_meta, lidar_points, ego_pose)
assert relations["edge_index"].shape[0] == 2
assert relations["distance_bins"].shape[0] == relations["edge_index"].shape[1]
assert relations["direction_bins"].shape[0] == relations["edge_index"].shape[1]
assert relations["topology_classes"].shape[0] == relations["edge_index"].shape[1]
assert relations["relation_mask"].dtype == torch.bool
assert relations["occlusion_mask"].dtype == torch.bool
```

### Phase 3：PSRD loss

产出：

- `tools/psrd_loss.py`
- `tests/test_psrd_loss.py`

目标：

```python
loss_dict = psrd_loss(camera_tokens, lidar_relations)
loss = loss_dict["loss_total"]
assert "loss_occ" in loss_dict
assert torch.isfinite(loss)
```

### Phase 4：robust training

产出：

- `modules/sensor_dropout.py`
- `tools/train_video_psrd.py`
- `tools/robust_eval.py`

目标：

- clean / drop / front-only / occlusion 评测表。

### Phase 5：VLM spatial QA adapter

产出：

- `modules/spatial_slot_reader.py`
- `modules/slot_hungarian_matcher.py`
- `modules/vlm_text_space.py`
- `tools/lgss_loss.py`
- `modules/vlm_projector.py`
- `dataset/driving_spatial_qa.py`
- `dataset/spatial_phrase_templates.py`
- `tools/train_vlm_adapter.py`
- `tools/eval_spatial_qa.py`

目标：

- frozen VLM。
- 只训练 adapter。
- 输出 spatial QA 分项结果。
- 输出 template split / paraphrase split 两套结果。
- 输出 no-LGSS / no-diversity / no-slot-temporal-consistency ablation。
- 输出 CLIP text encoder vs target-VLM text space ablation。

---

## 17. 写作时必须避免的坑

1. **不要说“我们解决了 autonomous driving VLM”**
   - 只说提升 spatial reasoning 和 robustness。

2. **不要把 LiDAR teacher 写成部署依赖**
   - 明确 LiDAR 只在训练阶段使用。

3. **不要再引入 VPR 任务**
   - 当前主线与 VPR 无关，论文叙事、实验表和 ablation 都不要放 VPR。

4. **不要只做 probing**
   - 必须做 VLM spatial QA，否则 VLM claim 不成立。

5. **不要堆 loss**
   - 方法核心必须围绕 PSRD。

6. **不要过度依赖自构造 QA**
   - 自构造 QA 可以做，但最好和 DriveLM / DriveSpatial / STSBench-style protocol 对齐。

7. **不要让 LGSS 变成模板记忆**
   - 必须有 paraphrase split、object-composition split 和 no-LGSS ablation。

8. **方法图必须画清楚 token 流**
   - `multi-view images → image patch tokens → calibration-aware cross-attention → BEV anchor tokens → PSRD relation graph loss → LGSS slots → frozen VLM adapter`。
   - 这条链路是方法核心，不要只画成抽象的 camera encoder 到 VLM。

---

## 18. 推荐摘要草稿

Camera-only driving VLMs have shown promising semantic and action-level capabilities, yet they remain weak in metric spatial reasoning, cross-view geometry, occlusion understanding, and robustness under sensor degradation. We propose GeoToken, a LiDAR-privileged relation pretraining framework that uses LiDAR only during training to make camera-only video tokens encode ego-centric spatial structure. Instead of regressing LiDAR features directly, GeoToken introduces Privileged Spatial Relation Distillation, which converts LiDAR observations into metric, directional, occupancy-topology, and occlusion-order relations and distills them into multi-view camera token interactions. To make these tokens accessible to language, we further use language-grounded spatial slots and evaluate them with a frozen driving VLM through a lightweight adapter. Experiments on nuScenes and KITTI-360 are designed to test driving spatial QA, hard paraphrase generalization, frozen depth/BEV/relation probing, and robustness under camera degradation and adverse conditions.

---

## 19. 当前优先级

最高优先级：

1. 先实现 `NuScenesClipDataset` 和 video tokenizer。
2. 再实现 LiDAR-to-camera patch projection。
3. 构建 PSRD relation label。
4. 做 depth / relation probing，确认 token 学到几何。
5. 最后接 frozen VLM adapter，做 spatial QA。

不建议现在投入：

- 训练完整大 VLM。
- 做完整 closed-loop。
- 同时上 nuScenes + KITTI-360 + Boreas。
- 一开始就换复杂 LiDAR teacher。

---

## 20. 会话日志

### 2026-05-17 — work3 Pro direction confirmed

- 确认采用“补 SimLingo / SpaceDrive 类纯视觉 VLM 模型空间几何短板”的方向。
- 明确核心方案：使用 LiDAR teacher 将三维空间关系蒸馏给 camera student。
- 初版文档记录了定位、动机、方法、实验、代码路线和 MVP。

### 2026-05-30 — direction narrowed for CVPR 2027

- 从“多头几何蒸馏 + VLM-ready tokens”的宽泛方案，收敛为 **LiDAR-Privileged Spatial Token Pretraining**。
- 将核心创新聚焦到 **Privileged Spatial Relation Distillation, PSRD**。
- 将主实验顺序调整为：VLM spatial QA → frozen spatial probing → robustness。
- 明确第一篇不训练大 VLM，不做完整 closed-loop，以 frozen VLM adapter 和 planning-relevant spatial QA 建立 VLM 价值闭环。

### 2026-05-31 — wording tightened around ego-centric relations and language access

- 将定位进一步收敛为 **LiDAR-privileged relation pretraining for language-accessible camera tokens**。
- 明确 camera tokens 不是字面意义的 ego-coordinate points，而是通过 LiDAR-derived ego-centric relations 学到 metric / topological / occlusion cues。
- 为 PSRD 补上 `L_occlusion_order`，避免遮挡评测和训练目标不闭环。
- 将 LGSS 定位为 language access mechanism，并加入 paraphrase split / object-composition split / no-LGSS ablation，降低模板记忆风险。

### 2026-05-31 — PSRD and LGSS implementation details hardened

- 将 PSRD 主图节点固定为 ego-centric BEV anchor tokens，不再混用 image patch / object region / BEV cell。
- 补充 distance / direction / topology / occlusion class 定义、`m_ij` 与 `m_occ_ij` 可靠性规则、`rho(o_ij)`、pair sampling 和默认 loss weights。
- 明确 pairwise 训练使用 sparse edge sampling，默认 `L=400`, `P=32`，复杂度从 `O(L^2)` 降为 `O(LP)`。
- 将 LGSS 拆成 object / region / relation slots，并补充 Hungarian matching、multi-positive contrastive、slot diversity loss 和 temporal slot consistency。

### 2026-05-31 — text encoder and active anchor ambiguity fixed

- 明确 `40 x 40 = 1600` 是 candidate BEV anchors，默认训练使用按优先级筛选的 `L=400` active anchors。
- 补充 active anchor 选择规则：3D box、occupied evidence、free-space boundary、camera-visible anchor、uniform free-space sample。
- 明确 LGSS text encoder 必须优先使用目标 frozen VLM 的 text pathway 或 adapter space；CLIP text encoder 只作为 CLIP-aligned VLM 默认或 ablation。
- 在 ablation 中加入 `CLIP text encoder vs target-VLM text space`。

### 2026-05-31 — method wording compressed and inference anchor selection clarified

- 将推理阶段 active-anchor selection 改成 camera-only：full 1600 anchors with chunked attention，或用 tokenizer 预测的 camera-frustum visibility / occupancy / objectness 选择 400 anchors。
- 压缩 LGSS text encoder 段落，保留 VLM-interface aligned text representation 的原则，避免写成 rebuttal。
- 统一术语：image patch tokens、BEV anchor tokens、spatial slots、adapter tokens。
- ablation 增加 mismatched text encoder 和 shuffled phrases。
- 明确主图需要画出 `image patch tokens → calibration-aware cross-attention → BEV anchor tokens → PSRD → LGSS → frozen VLM adapter`。
