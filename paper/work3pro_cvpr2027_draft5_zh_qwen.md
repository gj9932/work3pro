# GeoDistill-VLM: Distilling Privileged LiDAR Geometry into Native Vision-Language Tokens for Camera-Only Driving

匿名 CVPR 投稿

Paper ID：TBD

---

## 摘要

通用视觉语言模型具有强语义能力，但其视觉 token 缺少自动驾驶所需的可靠度量深度、跨视角三维对应与遮挡结构。现有空间增强方法通常在推理时额外运行单目深度网络，带来显存和延迟开销，并把外部深度误差直接写入 VLM 的三维位置编码。

我们提出 **GeoDistill-VLM**，在冻结 Qwen2.5-VL 原生视觉路径上将训练期 LiDAR 几何蒸馏到视觉 token。核心方法 **Native Token–LiDAR Fused Geometric Transport（NTL-FGT）** 将 Qwen 原生视觉 token 与 LiDAR 几何元素分别建模为带属性的离散测度，并通过 feature-level transport cost 与 structure-level Gromov-Wasserstein cost 学习 soft token-LiDAR coupling。该 coupling 的逐 token transport barycenter 提供 metric depth、三维坐标和可靠度 teacher，避免将单个硬投影或局部分位数直接视为唯一监督。

在 transport teacher 上，**Risk-Reliability Adaptive Companding（R²AC）** 继续将车辆动力学风险与视觉可预测的几何可靠度转化为 token 级连续压扩强度，使安全关键区域获得更高的深度监督灵敏度。适配器同时预测 sub-token ray offset、metric depth 和跨视角关系，并在 CKA 语义保持约束下，将三维位置编码与关系残差通过 zero-init gate 注入原生视觉 token。LiDAR、ego pose、标定和可选 3D 标注仅在训练期构造 transport、深度、可靠度与关系监督；部署时只使用多视角图像、标定和标准 ego state，不使用 LiDAR、最优传输求解器或外部深度估计器。我们在深度精度—效率权衡、驾驶空间问答、跨视角几何、nuScenes 开环规划、Bench2Drive 闭环驾驶和相机退化鲁棒性上评估该设计。

---

## 1. 引言

大规模视觉语言模型通过图文预训练获得了强大的语义理解和语言推理能力。以 Qwen2.5-VL [1] 为代表的通用 VLM 已能够识别复杂场景、理解文本指令并生成结构化回答，因此逐渐被用于自动驾驶场景描述、风险分析、反事实推理和轨迹规划。

然而，通用视觉语义并不等价于可靠的三维空间理解。自动驾驶 VLM 可能正确识别前方存在车辆，却无法稳定回答：

- 该车辆距离 ego 是 10 米还是 30 米；
- 左前方车辆与正前方车辆哪个更近；
- 同一车辆是否同时出现在前视和左前视相机中；
- 远处行人是否被前车遮挡；
- 右侧是否存在连续可通行区域；
- 给定候选轨迹是否会与某个目标相交。

这些问题要求视觉 token 同时携带语义、度量尺度和跨视角几何关系。Qwen2.5-VL 自带的 Vision Encoder 能够提供强语义表示，但其原始预训练目标并不保证 token 与真实三维坐标一一对应。

SpaceDrive [2] 表明，可以使用冻结深度估计器预测多视角 metric depth，将每个视觉 token 反投影到三维坐标，再把 universal 3D positional encoding 叠加到 VLM 的视觉 token 上。该设计的重要启发是：不必重新训练 Vision Encoder，空间信息可以在 Vision Encoder 和视觉语言 projector 之后注入。

但该方案存在两个限制。第一，推理阶段仍需要运行一个独立的稠密深度网络，增加计算和显存成本。第二，VLM 的空间表示直接依赖单目深度估计结果；在远距离、夜间、遮挡和跨域场景中，深度误差会被编码为错误的 3D PE。

另一方面，自动驾驶训练数据通常包含同步 LiDAR。即使部署端只使用相机，训练期 LiDAR 仍可提供准确的 metric geometry。这引出本文的问题：

> 能否冻结 Qwen2.5-VL 的原生 Vision Encoder 和 merger，仅训练一个轻量几何适配器，使其从 Qwen 原生视觉 token 中恢复可用于 VLM 推理的三维结构，并在推理时完全移除 LiDAR 和外部深度网络？

GeoDistill-VLM 为此设计了 Native Token–LiDAR Fused Geometric Transport 与 LiDAR-Privileged Geometry Adapter。多视角图像仍沿用 Qwen2.5-VL 的原始视觉路径：

```text
multi-view images
  -> frozen Qwen Vision Transformer
  -> frozen spatial merger
  -> image_embeds
```

训练时，我们先在冻结的 `image_embeds` 与 LiDAR 几何测度之间求解局部、熵正则的 fused geometric transport。其 soft coupling 同时考虑 token-LiDAR 的投影/射线一致性和 token-token 与 LiDAR-LiDAR 的成对结构，并通过逐 token barycentric projection 构造稳定的三维 teacher。随后，`image_embeds` 之后的轻量几何适配器完成两件事：

1. 从视觉 token 联合预测 R²AC 深度与压扩强度，恢复 token 对应的三维位置并生成 explicit 3D PE；
2. 从多视角 token 交互中学习 LiDAR 提供的跨视角、拓扑和遮挡关系，生成 geometry residual。

最终输入 LLM 的视觉 token 为：

```text
H_geo = H_img
      + alpha_pe  Phi(C_hat)
      + alpha_rel DeltaH_rel
```

其中 `H_img` 是冻结 Qwen merger 输出，`C_hat` 是轻量适配器预测的三维坐标，`Phi` 是 universal 3D positional encoder，`DeltaH_rel={Delta h_p^rel}` 是关系蒸馏得到的几何残差。两个门控系数均从零初始化，并使用 token-level CKA 约束增强前后的语义关系结构，使训练初始状态严格退化为原始 Qwen2.5-VL，并限制几何 residual 对原生 token manifold 的任意扭曲。

与独立 camera tokenizer 方案不同，GeoDistill-VLM 不产生另一套与 VLM 语义空间分离的 token，也不需要额外 projector 将自定义 token 映射到 LLM。它直接保留 Qwen 原有视觉语义，只学习原生 token 上的几何增量。

本文贡献如下：

1. 提出 **Native Token–LiDAR Fused Geometric Transport**，将训练期 LiDAR 几何测度与冻结 Qwen 原生视觉 token 建模为 feature-structure fused optimal transport 问题，通过 soft token-LiDAR coupling 和 transport barycenter 构造 token-level 几何 teacher，并在语义保持约束下实现最小扰动的几何注入。
2. 提出 R²AC 连续深度表示，将车辆动力学风险、局部几何一致性与 transport concentration 统一为 token 级压扩变量；该表示严格单调、解析可逆，并具有显式可控的近远场监督灵敏度比。
3. 提出面向多视角原生 VLM token 的稀疏关系蒸馏，通过标定感知边、跨视角可见性标签和置信度聚合学习相对位移、对应、拓扑与遮挡关系。

---

## 2. 相关工作

### 2.1 自动驾驶视觉语言模型

DriveLM [5] 以图结构视觉问答连接感知、预测与规划，DriveVLM [8] 和 OmniDrive [6] 强化复杂场景推理与规划，SimLingo [7] 则进一步评估视觉语言表征对闭环驾驶的作用。这些方法表明 VLM 能够利用大规模预训练语义与常识，但其空间能力通常仍依赖文本数字 token、额外 BEV 模块或任务特定查询。GeoDistill-VLM 关注更窄的问题：在不改变原始 Vision Encoder 的情况下，使其原生视觉 token 获得显式度量位置和场景关系。

### 2.2 空间感知 VLM

SpatialVLM [10]、LLaVA-3D [9] 和 SpaceDrive [2] 等方法通过空间问答、三维特征、位置编码或坐标接口提升 VLM 的空间能力。SpaceDrive 使用冻结深度估计器得到 metric depth，并将 3D PE 叠加到视觉 token 上。GeoDistill-VLM 沿用“在语言模型消费视觉 token 前注入空间信息”的思路，但空间来源不同：训练时以 LiDAR 为 privileged teacher，推理时由轻量 adapter 从 Qwen token 与标准 ego state 预测几何，不再运行外部深度模型。

### 2.3 LiDAR-to-Camera 特权学习

知识蒸馏 [30] 和 learning using privileged information（LUPI）[33] 允许训练阶段使用部署时不可用的教师信号。LiDAR-to-camera 方法通常对齐深度、BEV 特征、点特征、检测 logits 或 occupancy；UniDistill [31] 和 CMKD [32] 是其中具有代表性的跨模态蒸馏框架。多数方法服务于三维检测或 BEV 感知，并使用独立相机 backbone。GeoDistill-VLM 将 LiDAR 教师直接对齐到通用 VLM 的原生视觉 token，目标不是训练新的检测 backbone，而是改善语言模型可访问的 metric geometry。

特权信息并不保证能够被图像分支有效吸收，收益也可能来自结构变化或数据偏差 [34]。因此本文同时报告无 LiDAR、同等 LiDAR 标签预算的校准基线、冻结 token probe，以及 depth、allocation 和 relation label shuffle，避免仅凭最终任务提升宣称发生了几何知识迁移。

### 2.4 Optimal Transport 与结构对齐

最优传输通过 coupling 对齐两个离散测度，熵正则可使用 Sinkhorn 迭代高效近似 [36]。Gromov-Wasserstein 距离进一步比较两个测度内部的成对关系，而不要求两侧元素处于同一特征空间 [37]；Fused Gromov-Wasserstein 同时保留元素级 feature cost 和结构级关系 cost，适合对齐具有节点属性与图结构的异构对象 [38]。现有跨模态蒸馏通常直接匹配点、像素、BEV feature 或 logits。本文将冻结 Qwen 原生视觉 token 与训练期 LiDAR 几何元素建模为两个异构测度，以局部投影与射线约束 feature transport，以三维距离、可见性、表面与遮挡关系约束 structure transport，并使用 coupling 的逐 token barycentric projection 生成部署模型的几何 teacher。

### 2.5 单目度量深度

MiDaS [19,20] 强调跨数据集 relative depth，ZoeDepth [21]、Metric3Dv2 [24]、UniDepth [25]、UniDepthV2 [3] 和 Depth Anything V2 [23] 则进一步研究 metric depth 与泛化。Marigold [22] 使用扩散先验恢复单目深度。AdaBins [16] 和 LocalBins [17] 通过图像或局部分布预测离散深度区间，适合稠密深度解码，但需要多个 bin center 与概率，并且其局部灵敏度不能由单个物理量直接解释。固定对数压扩也广泛用于信号编码，例如 G.711 [18]；Vision Banana [11] 则采用 Barron 的固定 power transform [15] 与可逆 RGB 编码，从生成模型输出中恢复 metric depth。R²AC 保持单标量连续回归和解析反变换，同时把压扩强度定义为可监督的 token 几何变量；其近远场灵敏度比可由闭式公式直接控制，适合后续反投影与 3D positional encoding。

### 2.6 多视角三维感知与规划

PETR [27] 通过三维位置编码组织多视角图像特征，BEVFormer [26] 以时空注意力构建 BEV 表示，UniAD [28] 和 VAD [29] 则将感知、预测与规划统一到端到端驾驶框架中。这些方法主要优化任务专用 BEV query 或规划 token。GeoDistill-VLM 不试图替代完整 BEV 感知栈，而是研究一个互补问题：能否在冻结通用 VLM 视觉路径的条件下，把稀疏度量几何写入语言模型实际消费的原生视觉 token。

---

## 3. 方法

### 3.1 问题定义

训练阶段样本包含：

```text
X_cam   = {I_t,n},  t = 1..T, n = 1..N_cam
X_lidar = {P_t},    t = 1..T
K       = {K_n}
T_cam   = {T_ego->cam,n}
T_ego   = {T_t->t0}
S_ego   = {v_ego, optional yaw_rate}
```

其中 `I_t,n` 为多视角图像，`P_t` 为 LiDAR 点云，`K_n` 和 `T_ego->cam,n` 为相机内外参，`T_t->t0` 用于将相邻帧对齐到中心帧，`S_ego` 为车辆可直接读取的运动状态。R²AC 主方法只要求 ego speed，yaw rate 用于可选的弯道走廊扩展。

推理阶段输入为：

```text
X_infer = {X_cam, K, T_cam, S_ego}
```

LiDAR、3D box 和地图标签仅用于训练监督，不进入部署路径。推理使用相机、标定和车辆自身状态，不使用额外环境传感器。

目标模型基于 `Qwen2.5-VL-7B-Instruct`。默认视觉配置为：

| 模块 | 配置 |
|---|---|
| Vision Encoder | `Qwen2_5_VisionTransformerPretrainedModel` |
| ViT 层数 | 32 |
| Patch size | 14 |
| ViT hidden size | 1280 |
| Attention heads | 16 |
| Window size | 112 |
| Spatial merge | 2 × 2 |
| Merger 输出维度 | 3584 |

我们冻结：

- Qwen Vision Encoder；
- Qwen spatial merger / vision-language projector；
- 原始 token embedding 中与本方法无关的参数。

我们训练：

- 训练期 FGT feature/structure cost adapters；
- LiDAR-Privileged Geometry Adapter；
- 3D PE 门控和 relation residual 门控；
- 可选 coordinate decoder / planning head；
- Qwen LLM 的 rank-16 LoRA。

### 3.2 冻结 Qwen2.5-VL 视觉路径

每个相机图像使用 Qwen2.5-VL 原始图像预处理和动态网格描述 `image_grid_thw`。视觉编码器首先产生 patch feature，再由原生 merger 完成 `2 x 2` spatial merge 并对齐到 LLM hidden space：

```text
X_vis = f_visual(X_cam, image_grid_thw)
H_img = g_merger(X_vis)
```

对应 HuggingFace 模型接口：

```python
image_embeds = model.get_image_features(
    pixel_values,
    image_grid_thw,
)
```

`image_embeds` 已包含 Qwen visual transformer 和 merger 的输出。

其中：

```text
H_img = {h_p}_(p=1..N),  h_p in R^3584
```

对于六相机输入，各视角 token 按已知边界拼接。我们记录每个 merged token 的：

```text
meta_p = {
  camera_id,
  frame_id,
  image_grid_thw,
  token_center_uv,
  token_region,
  K,
  T_ego->cam
}
```

GeoDistill-VLM 不修改 `f_visual` 和 `g_merger`，也不额外增加从自定义视觉特征到 LLM 的 projector。所有几何模块直接读取 `H_img`。

### 3.3 LiDAR-Privileged Geometry Adapter

LPGA 包含五个轻量组件；其监督 teacher 由训练期 NTL-FGT 构造：

1. **Token geometry encoder**：融合 Qwen image token、camera ID 和归一化图像位置；
2. **R²AC depth head**：预测每个 merged token 的连续压扩深度；
3. **Factorized allocation head**：分别预测驾驶风险与视觉几何一致性，并组合为 token 级压扩强度；
4. **Sub-token ray head**：预测 token 区域内代表表面的二维射线偏移；
5. **Relation adapter**：融合标定信息，在稀疏跨 token 图上聚合三维关系并输出 geometry residual。

对 token `p`：

```text
g_p^mono = A_token([
  LN(h_p),
  e_cam(p),
  e_uv(p)
])

g_p^geo = A_geo([
  g_p^mono,
  e_calib(p)
])
```

其中 `e_ego` 是对 `S_ego` 中 ego speed 与可选 yaw rate 的 sinusoidal embedding，并在风险分支中与 token 特征沿通道维拼接。`A_token` 为小型 MLP 或 2 层 Transformer，输出维度远小于 LLM hidden size。深度、allocation 与 ray head 只读取 `g_p^mono`；风险 head 额外读取 `e_ego`，标定仅进入反投影与 relation adapter，避免直接利用相机参数记忆数据集深度先验。默认使用 bottleneck：

```text
3584 -> 512 -> 512
```

LPGA 不复制完整视觉 backbone，其参数量应显著小于 Qwen Vision Encoder。

### 3.4 LiDAR 几何元素与硬标签基线

#### LiDAR 投影

将 LiDAR 点转换到中心 ego 坐标，再投影到各相机：

```text
p_cam = T_ego->cam p_ego
[u, v, 1]^T ~ K p_cam
```

仅保留：

- 位于相机前方的点；
- 落入图像范围的点；
- 通过时间同步和基础可见性检查的点；
- 未被明显动态错位污染的点。

由于 Qwen merger 合并 `2 x 2` patch，我们直接在 merged token 的图像区域 `R_p` 内聚合 LiDAR 点。对每个 token：

```text
Q_p = {
  (u_j, v_j, d_j, c_j^ego)
  | projected point j in R_p
}

D_p = {d_j | j in Q_p}
```

若 `D_p` 非空，原始 hard baseline 使用低分位数而不是均值：

```text
d_p^Q = Quantile(D_p, q=0.1)
```

该策略保留前景表面，同时比单个最小值对离群点更稳健。令 `m_p^Q=1[D_p != empty]` 表示 token 具有有效 hard depth label；`min depth` 作为与 SpaceDrive 对齐的消融。主方法不再把 `d_p^Q` 视为唯一 teacher，而是在第 3.5 节通过 soft coupling 构造 transport barycenter depth；`d_p^Q` 仅作为 hard-assignment 基线和可选 warm-up 辅助目标。

为避免把区域深度错误地放在 token 中心射线上，先选取接近前景分位数的 LiDAR 子集：

```text
S_p^Q = {j in Q_p | d_j <= d_p^Q + tau_fg}

(u_p^Q, v_p^Q) =
  sum_(j in S_p^Q) omega_j (u_j, v_j)
  / (sum_(j in S_p^Q) omega_j + epsilon)

delta_p^Q = [
  2 (u_p^Q - u_p) / width(R_p),
  2 (v_p^Q - v_p) / height(R_p)
]
```

其中 `omega_j` 由时间距离和投影置信度确定，`delta_p^Q` 截断到 `[-1,1]^2`。该 hard offset 在消融中与 transport barycenter ray teacher 比较。

LiDAR 点数反映传感器采样密度，不要求图像分支在推理时预测。我们将其与可由图像边界和局部结构近似判断的几何一致性分开：

```text
s_p = 1 - exp(-n_p / n_0)

q_p^geom =
  exp(
    - MAD(D_p)
    / (tau_q Median(D_p) + epsilon)
  )

q_p^label = s_p q_p^geom
```

其中 `n_p=|D_p|`，`s_p` 仅表示 hard label 的采样支持度，`q_p^geom` 表示区域内是否接近单一表面，`q_p^label` 是 hard-label reliability；当 `D_p` 为空时定义 `q_p^geom=q_p^label=0`。allocation head 不预测 LiDAR 点数或采样模式。`s_p` 同时用于可靠度监督，因此单点 token 即使因 `MAD=0` 得到 `q_p^geom=1`，也不会获得满置信度。

#### 多帧 LiDAR 增强

训练时可将相邻 LiDAR sweep 通过 ego pose 对齐到中心帧，提高远距离覆盖。动态 box 内的点只使用中心帧，避免运动目标产生重影：

```text
P_static = EgoMotionCompensate(P_t)
P_dynamic = P_center within dynamic boxes
```

多帧累积只用于构造更密集的训练标签，不改变推理输入。

### 3.5 Native Token–LiDAR Fused Geometric Transport

Hard projection 将一个 token 区域压缩为单个分位数，无法表达 merged token 覆盖多个表面时的分配不确定性，也没有利用 token-token 与 LiDAR-LiDAR 的成对结构。我们将冻结 Qwen 原生 token 与训练期 LiDAR teacher 分别表示为离散测度，并在二者之间求解 feature-structure fused transport。该过程只用于构造监督，不进入推理路径。

#### Native token 与 LiDAR 几何测度

冻结 Qwen merger 输出为：

```text
H_img = {h_p}_(p=1..N),  h_p in R^D
```

LiDAR teacher 几何元素为：

```text
G_L = {(c_i, a_i^L)}_(i=1..M)
```

其中 `c_i in R^3` 是中心 ego 坐标系中的 LiDAR 点、voxel center 或局部 surface anchor，`a_i^L` 可包含 depth、camera visibility、object ID、occupancy、timestamp confidence 和 surface/voxel ID。默认实现先对多帧点云做动态目标过滤和体素/表面聚合，以控制 transport 规模。

定义 token-side measure 与 LiDAR-side geometric measure：

```text
mu_V = sum_p m_p^V delta(h_p)
mu_L = sum_i m_i^L delta(c_i)
```

`m_p^V` 由有效 token mask、相机可见区域和可选 foreground prior 归一化得到；`m_i^L` 由时间同步、投影质量和 LiDAR anchor confidence 归一化得到。二者均为非负质量。我们学习 transport coupling：

```text
pi in R_+^(N x M)

pi 1_M   = m^V
pi^T 1_N = m^L
```

全局 balanced coupling 适合小规模 anchor。默认实现只在投影邻域、相邻 ray bin 和重叠相机候选内匹配，并使用 partial OT 或 relaxed marginal：

```text
pi 1_M   <= m^V
pi^T 1_N <= m^L

sum_(p,i) pi_(p,i) = m_tr
```

其中 `0 < m_tr <= min(||m^V||_1,||m^L||_1)` 是每个局部问题或 batch 的 transported-mass budget。该约束排除 `pi=0` 的退化解；未被可靠解释的剩余 token 或 LiDAR anchor 仍可保留未传输质量，避免强迫天空、反光或严重遮挡区域产生错误对应。也可使用带 marginal KL penalty 的 unbalanced OT，但主方法固定 `m_tr` 以减少额外自由度。

#### Feature-level transport cost

令 `bold u_p=(u_p,v_p)` 为 token region center 或当前预测的 sub-token ray 像素位置，`Pi_n(c_i)` 为 `c_i` 在 token `p` 所属相机中的投影，`rho_p` 为 token ray direction，`rho_i` 为该 LiDAR anchor 在同一相机下的 ray direction。元素级 cost 为：

```text
M_(p,i) =
    lambda_uv  ||bold u_p - Pi_camera(p)(c_i)||_2^2
  + lambda_ray (1 - <rho_p, rho_i>)
  + lambda_cam 1[c_i not visible in camera(p)]
  + lambda_d |d_p^prior - d_i|
```

其中 `d_i` 是 `c_i` 的相机深度。`d_p^prior` 是可选项：allocation warm-up 时令 `lambda_d=0`；深度 head 稳定后，可使用 `StopGrad(d_hat_p)` 或不依赖 LiDAR 的 coarse depth prior。停止梯度避免 student 通过改变 prior 操纵 coupling。投影范围、正深度和稀疏 z-buffer visibility 作为 hard candidate mask，`lambda_cam` 只对剩余候选施加软惩罚。

#### Structure-level Gromov-Wasserstein cost

元素级投影仍可能在重复纹理、跨视角重叠和稀疏远场产生歧义。为此定义 token-side structure：

```text
C_pq^V =
  A_struct^V([
    LN(h_p), LN(h_q),
    e_cam(p), e_cam(q),
    e_uv(p), e_uv(q),
    xi_pq
  ])
```

其中 `xi_pq` 是第 3.10 节使用的标定感知相对 ray 特征。实现上也可复用 relation adapter 的 edge embedding：

```text
C_pq^V = A_struct^V(r_pq)
```

LiDAR-side structure 为：

```text
C_ij^L = [
  ||c_i - c_j||_2,
  sign(d_j - d_i),
  1[same object],
  1[same voxel or surface],
  visibility_ij,
  occlusion_ij
]
```

`A_struct^V` 输出与上述 LiDAR vector 同语义的连续预测和分类 logits；连续 LiDAR 量按数据集尺度归一化，离散量使用 one-hot 或 soft confidence。令 `K_cont` 与 `K_disc` 分别表示连续和离散结构分量，定义：

```text
D_struct(C_pq^V, C_ij^L) =
  sum_(k in K_cont)
    lambda_k Huber(
      C_pq^V[k],
      Normalize(C_ij^L[k])
    )
  + sum_(k in K_disc)
    lambda_k CE(
      C_pq^V[k],
      C_ij^L[k]
    )
```

该 typed structure difference 不要求 `h_p` 与 `c_i` 位于同一特征空间，只要求被 coupling 对齐的 token 对与 LiDAR anchor 对具有一致的相对几何。LiDAR target 侧不使用可学习投影，避免两侧投影共同塌缩为常量。

#### Fused geometric transport objective

最终的 Fused Gromov-Wasserstein distillation loss 为：

```text
L_FGT =
  (1 - gamma) sum_(p,i) M_(p,i) pi_(p,i)
  + gamma sum_(p,q,i,j)
      D_struct(C_pq^V, C_ij^L)
      pi_(p,i) pi_(q,j)
  + epsilon_ot sum_(p,i)
      pi_(p,i) (log(pi_(p,i) + epsilon) - 1)
```

第一项对齐 token-LiDAR 的局部投影、射线、可见性和可选深度 prior；第二项对齐 token-token 与 LiDAR-LiDAR 的成对几何结构；第三项为 entropic regularization，用于平滑 coupling 并稳定优化。`gamma in [0,1]` 控制 feature alignment 与 structure alignment 的权衡，`gamma=0` 退化为只使用元素级 cost 的熵正则 OT。

实际训练不显式枚举 `O(N^2 M^2)` 四元组。我们在每个 token 的局部 LiDAR 候选和第 3.10 节的稀疏 token edge 上交替线性化 GW 项，并用 Sinkhorn/proximal 更新 partial coupling；结构项使用采样到的可靠边估计。FGT 仅在训练期运行，推理时不保留 coupling、LiDAR anchor 或 OT solver。

#### Transport barycenter teacher

对 token `p`，使用 coupling 的行归一化 barycentric projection 构造三维 teacher：

```text
bar c_p^L =
  sum_i StopGrad(pi_(p,i)) c_i
  / (sum_i StopGrad(pi_(p,i)) + epsilon)

bar d_p^L =
  ||T_ego->cam(p) bar c_p^L||_z
```

其中 `||.||_z` 表示相机坐标系中的正向 `z` 深度。令 `I_p` 为 token `p` 的有效 LiDAR 候选集合，并定义：

```text
tilde pi_(p,i) =
  pi_(p,i)
  / (sum_(k in I_p) pi_(p,k) + epsilon)

q_p^conc =
  1 - H(tilde pi_(p,:)) / log(|I_p|)

q_p^mass =
  clip(
    sum_i pi_(p,i) / (m_p^V + epsilon),
    0, 1
  )

q_p^OT = q_p^conc q_p^mass
```

当 `|I_p|=1` 时定义 `q_p^conc=1`。`q_p^conc` 衡量分配是否集中，`q_p^mass` 衡量 partial OT 中有多少 token 质量获得可靠解释，因此 `q_p^OT in [0,1]` 同时反映 coupling 的确定性与覆盖率。主方法将局部表面一致性与 transport reliability 融合：

```text
eta_(q,p) = eta_q s_p

q_p^teacher =
  eta_(q,p) q_p^geom
  + (1 - eta_(q,p)) q_p^OT
```

`s_p` 使单点或低支持度 hard statistics 不会主导融合；当 token 内没有可用 hard region statistics 时，`eta_(q,p)=0`，可靠度完全由 transport concentration 与 transported mass 决定。

并定义主 teacher：

```text
d_p^T = bar d_p^L
c_p^T = bar c_p^L

m_p^T =
  1[
    sum_i pi_(p,i) / (m_p^V + epsilon)
    > tau_mass
  ]
  1[d_p^T >= 0]
```

将 `bar c_p^L` 投影回 token 相机可得到 transport ray teacher：

```text
(bar u_p^L, bar v_p^L) =
  Pi_camera(p)(bar c_p^L)

delta_p^T = [
  2 (bar u_p^L - u_p) / width(R_p),
  2 (bar v_p^L - v_p) / height(R_p)
]
```

后续 R²AC、ray 和 coordinate supervision 默认使用 `(d_p^T,c_p^T,delta_p^T,q_p^teacher)`。Quantile teacher `(d_p^Q,delta_p^Q)` 仅用于 hard projection 对照、无结构 warm-up 或 coupling 失败时的可选 fallback。`StopGrad(pi)` 使 depth/coordinate heads 不能通过改变 coupling 降低自身监督误差；`L_FGT` 仍独立优化 transport cost 与 structure adapter。

### 3.6 Risk-Reliability Adaptive Companding Depth

固定 power-warp 对所有 token 使用相同的近场偏置，但驾驶场景中的监督需求并不均匀：位于制动走廊内的近距离目标比远处天空或静态背景更安全关键，稀疏或 transport 分配分散的 LiDAR teacher 也不应获得与高可靠 teacher 相同的监督强度。为此，我们提出 **Risk-Reliability Adaptive Companding（R²AC）**，联合学习压扩深度 `zeta_p` 与 token 级灵敏度变量 `a_p`。FGT 负责构造稳健的 token-level teacher，R²AC 负责控制该 teacher 在深度域中的监督灵敏度，二者职责不同且联合训练。

#### 驾驶风险与灵敏度分配目标

根据 ego 速度构造可解释的停车距离：

```text
d_safe =
  v_ego t_react
  + v_ego^2 / (2 a_brake)
  + d_margin
```

该式采用常减速度停车距离，并遵循 RSS 类安全距离建模强调的可解释物理约束 [35]；它是风险监督的可控近似，而不是对真实交互风险的完整刻画。

令 `x_p^ego` 和 `y_p^ego` 分别为 transport barycenter `c_p^T` 在 ego 坐标系中的前向与横向距离，定义软风险：

```text
r_p =
  sigmoid((d_safe - x_p^ego) / tau_r)
  sigmoid(x_p^ego / tau_front)
  exp(-(y_p^ego)^2 / (2 w_corridor^2))
```

前两项选择 ego 前方且位于停车距离内的目标，第三项强调 ego 行驶走廊。令 `a_p in [0,1]` 表示压扩强度，结合第 3.5 节的 fused teacher reliability，oracle 监督为：

```text
a_p^* =
  r_p [eta + (1 - eta) q_p^teacher]
```

由于 `r_p,q_p^teacher in [0,1]`，上式天然满足 `a_p^* in [0,1]`。`eta > 0` 保留高风险但 transport 不确定 token 的最低近场偏置。为明确部署时压扩强度的来源，我们分别预测风险和 teacher reliability：

```text
r_hat_p = sigmoid(A_risk([g_p^mono, e_ego]))
q_hat_p = sigmoid(A_reli(g_p^mono))

a_hat_p =
  r_hat_p [eta + (1 - eta) q_hat_p]
```

`q_hat_p` 以 `q_p^teacher` 为监督，学习图像中的边界、遮挡、局部多表面和跨 token 结构歧义，而不是复现 LiDAR 采样密度。推理时 `r_hat_p`、`q_hat_p` 和 `a_hat_p` 完全由图像 token 与 ego state 预测，不使用 LiDAR、coupling 或可靠度标签。

#### 连续可逆压扩

设任务有效深度范围为 `d in [0, D_max]`，归一化深度 `x=d/D_max`，并定义绝对深度有效 mask：

```text
m_p^D =
  m_p^T 1[d_p^T <= D_max]
```

超过 `D_max` 的标签不参与绝对深度、坐标和排序损失，但仍可参与关系监督。定义：

```text
mu(a) = exp(beta a) - 1

F(d; a) =
  log(1 + mu(a) d / D_max)
  / log(1 + mu(a))
```

当 `a -> 0` 时，对分子与分母应用 L'Hôpital 法则得到连续极限：

```text
F(d; 0) = d / D_max
```

因此低风险 token 自动退化为均匀线性深度；`a` 增大时，同样的一米近场深度变化会在压扩目标中产生更大变化，从而提高近场监督灵敏度。其解析逆变换为：

```text
F_inv(zeta; a) =
  D_max
  [exp(beta a zeta) - 1]
  / [exp(beta a) - 1]
```

当 `a=0` 时，`F_inv(zeta;0)=D_max zeta`。实现中对小于阈值的 `a` 使用上述极限，并分别使用 `log1p` 和 `expm1` 保证数值连续。这里使用 `zeta` 表示压扩域标量，避免与二维像素坐标 `(u,v)` 和三维坐标 `z` 冲突。

R²AC 对深度严格单调：

```text
dF / dd =
  mu(a)
  / {
      D_max log(1 + mu(a))
      [1 + mu(a) d / D_max]
    }
  > 0
```

其近端与最远端的灵敏度比具有直接解释：

```text
(dF/dd at d=0)
---------------- = 1 + mu(a) = exp(beta a)
(dF/dd at d=D_max)
```

因此 `a_p` 直接控制 token 的近远场监督灵敏度比，而不是不可解释的自由参数。默认 `D_max=80m`、`beta=log(16)`，使最大灵敏度比为 16；具体范围随数据集传感器有效距离设置。

#### 深度预测与防退化目标

为避免深度目标随 allocation 预测漂移，压扩域监督始终由固定 oracle `a_p^*` 构造；预测强度只用于 metric 解码和部署：

```text
zeta_hat_p = sigmoid(A_depth(g_p^mono))
zeta_p^*   = F(d_p^T; a_p^*)
d_hat_p    = F_inv(zeta_hat_p; StopGrad(a_hat_p))
```

对有效 transport teacher token，使用 fused reliability 加权。风险只通过 `a_p^*` 改变压扩灵敏度，避免与显式 loss reweighting 双重强化：

```text
w_p = m_p^D (q_p^teacher + epsilon_q)

L_comp =
  sum_p w_p SmoothL1(
    zeta_hat_p,
    zeta_p^*
  )
  -------------------------------
       sum_p w_p + epsilon

L_r =
  sum_p m_p^D SmoothL1(r_hat_p, r_p)
  ---------------------------------
          sum_p m_p^D + epsilon

L_q =
  sum_p m_p^D q_p^mass
    SmoothL1(q_hat_p, q_p^teacher)
  ----------------------------------
    sum_p m_p^D q_p^mass + epsilon

L_alloc =
  lambda_r L_r
  + lambda_q L_q

lambda_r + lambda_q = 1
```

其中 `lambda_r` 与 `lambda_q` 只表示两个因子的相对权重并归一化为和为 1，外层 `lambda_alloc` 控制 allocation 模块相对 metric depth 的总权重，避免三者形成冗余自由度。`a_hat_p` 是 `r_hat_p` 与 `q_hat_p` 的解析组合，因此主方法不再对 `a_hat_p` 叠加冗余回归损失；当两个因子分别拟合其目标时，`a_hat_p` 自动趋近 `a_p^*`。仅在消融中额外加入：

```text
L_a^aux =
  sum_p m_p^D SmoothL1(a_hat_p, a_p^*)
  --------------------------------------
           sum_p m_p^D + epsilon
```

该消融用于检验组合目标的联合梯度是否损害两个因子的独立校准。`q_p^mass` 控制 partial transport teacher 的监督覆盖度，风险分支使用所有具有有效 barycenter 的 token。`StopGrad` 阻止 allocation head 通过改变逆变换来吸收 metric depth 误差；allocation 只能通过 `L_alloc` 学习可解释因子。由于不同 token 使用不同压扩强度，不能直接比较其 `zeta_hat`。因此排序损失作用于解码后的 metric depth：

```text
s_pq = sign(d_q^T - d_p^T)

L_rank =
  sum_pq w_pq log(
    1 + exp(
      -s_pq (d_hat_q - d_hat_p) / tau_d
    )
  )
  -----------------------------------------
           sum_pq w_pq + epsilon

w_pq = m_pq^D sqrt(w_p w_q)
m_pq^D = m_pq m_p^D m_q^D
```

其中 `m_pq` 表示在同一相机局部邻域或可靠跨视角对应中采样到的有效排序对。

定义 metric depth 目标：

```text
L_depth =
  L_comp
  + lambda_rank L_rank
```

`L_ray` 与 `L_coord` 在下一节定义，并与 `L_depth`、`L_alloc` 和 `L_rel` 共同构成完整 token 几何监督。

### 3.7 从预测深度恢复 Token 级三维位置

ray head 预测归一化 sub-token offset：

```text
delta_hat_p =
  tanh(A_ray(g_p^mono))

u_bar_p =
  u_p + width(R_p) delta_hat_p^u / 2

v_bar_p =
  v_p + height(R_p) delta_hat_p^v / 2
```

其监督目标为第 3.5 节由 transport barycenter 投影得到的 `delta_p^T`：

```text
L_ray =
  sum_p w_p SmoothL1(delta_hat_p, delta_p^T)
  -------------------------------------------
              sum_p w_p + epsilon
```

对预测射线 `(u_bar_p,v_bar_p)` 和深度 `d_hat_p`，使用相机模型反投影：

```text
c_hat_p^cam =
  d_hat_p K^-1 [u_bar_p, v_bar_p, 1]^T
```

再转换到中心 ego 坐标：

```text
c_hat_p^ego =
  T_cam->ego c_hat_p^cam
```

得到：

```text
C_hat = {c_hat_p^ego},  c_hat_p^ego in R^3
```

训练时，对存在有效 transport barycenter 的 token 直接计算坐标误差：

```text
L_coord =
  sum_p w_p Huber(c_hat_p^ego, StopGrad(c_p^T))
  ------------------------------------------------
            sum_p w_p + epsilon
```

因此 student 同时对齐 barycenter depth、barycenter ray 与三维位置。固定 token center、hard quantile 占用质心、transport barycenter ray 和预测 ray offset 将在消融中比较。

### 3.8 Universal 3D Positional Encoding

我们使用维度与 Qwen LLM hidden size 一致的三维 sine-cosine positional encoding：

```text
Phi(c_p) = [
  phi_x(x_p),
  phi_y(y_p),
  phi_z(z_p)
] in R^3584
```

对空间维度 `a in {x,y,z}`：

```text
phi_a(a)[2i]   = sin(a / tau^(2i/d_a))
phi_a(a)[2i+1] = cos(a / tau^(2i/d_a))
```

具体设置 `d_x=1194`、`d_y=1194`、`d_z=1196`，三者均为偶数且总和为 3584，因此可直接拼接而无需 padding。每个水平轴包含 597 组正余弦频率，垂直轴包含 598 组；`tau` 为频率尺度，并在有效坐标范围上进行灵敏度消融。

使用预测坐标得到：

```text
P_hat = {Phi(c_hat_p^ego)}
```

为避免破坏 Qwen 原始视觉 token 分布，3D PE 使用可学习标量门控：

```text
alpha_pe = alpha_pe_max * tanh(beta_pe)
beta_pe  = 0 at initialization
```

因此 `alpha_pe` 在初始化时严格为 0。这样训练开始时：

```text
alpha_pe Phi(c_hat_p^ego) = 0
```

即 PE 分支在初始化时不改变原始视觉 token；relation gate 与 `W_up` 的初始化统一在第 3.11 节说明。

### 3.9 LiDAR 特权关系图

仅有 token depth 仍不足以表达跨视角对应、自由空间拓扑和遮挡关系。我们在 Qwen merged visual token 上构建稀疏 LiDAR teacher graph：

```text
G_lidar = (V_token, E_geo)
```

节点与 Qwen `image_embeds` 一一对应，不引入独立 BEV anchor token。

每个有效节点保存：

```text
v_p = {
  d_p^T,
  c_p^T,
  q_p^teacher,
  camera_id,
  occupancy_label,
  object_id optional
}
```

每条边保存：

```text
e_pq = {
  delta_c_pq,
  depth_order_pq,
  cross_view_pq,
  topology_pq,
  occlusion_pq,
  reliability_mask
}
```

其中：

- `delta_c_pq = c_q^T - c_p^T`；
- `depth_order` 表示同视角或相邻 ray 上的前后顺序；
- `cross_view` 表示两个 token 是否对应同一 3D voxel、表面或对象；
- `topology` 表示同一自由空间、同一占用区域或不同组件；
- `occlusion` 表示可靠 ray neighborhood 内的遮挡顺序。

与固定 BEV anchor 不同，这些 token 的三维位置由图像内容对应的 LiDAR 深度决定，不能仅通过 token 索引或二维坐标求得。

#### 跨视角标签构造

对每个 token 定义高质量 transport anchor 集合 `A_p^T={i | pi_(p,i)>tau_pi}`，并将其中 LiDAR anchor 保持在中心 ego 坐标。跨相机候选边仅从相邻相机重叠视场和三维近邻中产生。若两个 token 满足以下任一条件，则标记为 `cross_view=1`：

- 前景点落入同一 ego voxel，且代表深度差小于 `tau_depth`；
- 前景点关联到同一 3D object ID，且两个相机中均通过可见性检查。

可见性通过每个相机的稀疏 z-buffer 检查：候选点投影深度必须与该像素邻域的最近可靠深度一致。外观相似但 voxel、object ID 或可见性不一致的候选构成 hard negative。边可靠度定义为两个节点 `q_p^teacher`、`q_q^teacher`、coupling overlap、体素重叠率与深度一致性的乘积，并用于过滤关系损失。FGT 的 structure term 约束 teacher coupling 的成对一致性，`L_rel` 则显式训练推理期 relation head，二者互补而非替代。训练时使用 LiDAR/3D box 构造标签；推理时不构图标签，只在图像局部边和预测三维位置的 kNN 边上运行 relation adapter。

### 3.10 Sparse Relation Adapter

Relation Adapter 读取包含标定信息的 token geometry feature `g_p^geo`。二维坐标差仅对同相机边有意义，因此定义标定感知边特征：

```text
rho_p^ego =
  R_cam->ego normalize(
    K^-1 [u_bar_p, v_bar_p, 1]^T
  )

xi_pq =
  if camera(p) == camera(q):
    [delta_u / W, delta_v / H,
     rho_q^ego - rho_p^ego]
  else:
    [rho_p^ego, rho_q^ego,
     rho_q^ego - rho_p^ego,
     camera_pair_id]
```

两类特征维度和语义不同，因此分别使用输出维度相同但参数不共享的 edge encoder：

```text
r_pq =
  if camera(p) == camera(q):
    A_edge^same([
      g_p^geo, g_q^geo,
      g_p^geo - g_q^geo,
      g_p^geo * g_q^geo,
      xi_pq^same
    ])
  else:
    A_edge^cross([
      g_p^geo, g_q^geo,
      g_p^geo - g_q^geo,
      g_p^geo * g_q^geo,
      xi_pq^cross
    ])
```

`camera_pair_id` 在跨相机分支中使用可学习 embedding。同相机分支不接收无意义的 camera-pair 特征，跨相机分支也不使用两个独立像平面上的 `delta_u,delta_v`。

预测：

```text
delta_c_hat_pq,
depth_order_hat_pq,
cross_view_hat_pq,
topology_hat_pq,
occlusion_hat_pq
```

损失为：

```text
L_delta = Huber(delta_c_hat_pq, delta_c_pq)
L_order = CE(depth_order_hat_pq, depth_order_pq)
L_cross = BCE(cross_view_hat_pq, cross_view_pq)
L_topo  = CE(topology_hat_pq, topology_pq)
L_occ   = CE(occlusion_hat_pq, occlusion_pq)
```

仅对 reliability mask 有效的边计算损失：

```text
L_rel =
  lambda_delta L_delta
+ lambda_order L_order
+ lambda_cross L_cross
+ lambda_topo  L_topo
+ lambda_occ   L_occ
```

第一阶段默认启用：

```text
L_delta + L_order + L_cross
```

`L_topo` 和 `L_occ` 仅在 LiDAR/地图标签与 ray 可视化通过质量检查后启用。

#### Sparse edge sampling

每个 token 最多采样 `P` 条边：

- `P_local`：同视角相邻 token；
- `P_ray`：相同或相邻 ray bin；
- `P_cross`：跨相机同 voxel / 同对象正样本；
- `P_hard`：外观相似但三维位置不同的负样本；
- `P_far`：随机远距离有效 token。

训练复杂度从 `O(N^2)` 降为 `O(NP)`。

### 3.11 Semantic-Preserving Geometry Residual 注入

Relation Adapter 使用置信度归一化加权和将稀疏关系聚合回 token：

```text
ell_pq = A_conf(r_pq)

omega_pq =
  softmax_(q in N(p))(
    ell_pq + log(e_conf_pq + epsilon)
  )

z_p^rel =
  sum_(q in N(p))
    omega_pq V_rel r_pq
```

其中训练期 `e_conf_pq` 使用 teacher edge reliability，推理期使用 relation head 的预测置信度。mean aggregation 和无 teacher confidence 的 attention 作为消融。

再映射到 Qwen hidden size：

```text
Delta h_p^rel = W_up z_p^rel
```

最终输入 LLM 的视觉 token 为：

```text
h_tilde_p =
  h_p
  + alpha_pe  Phi(c_hat_p^ego)
  + alpha_rel Delta h_p^rel
```

其中：

```text
alpha_rel = alpha_rel_max * tanh(beta_rel)
beta_rel  = 0 at initialization
```

`W_up` 使用小方差非零初始化 `N(0, sigma_up^2)`，默认 `sigma_up=1e-4`。若 `W_up` 与 `alpha_rel` 同时精确为零，二者梯度都会消失；因此这里只将 relation gate 严格初始化为零，并保持投影非零。该设计使初始前向仍严格退化为原始 Qwen，同时允许 gate 在第一步获得梯度。

令 `H_geo={h_tilde_p}`。原有 `L_keep` 只约束全局 pooled representation，不能阻止局部 token relation 被任意重排。我们进一步使用 linear CKA [39] 保持原生 token 的关系结构。对去除 padding 后的每个相机 token matrix，令 `J=I-11^T/N_cam_token` 为中心化矩阵：

```text
H_geo^c = J H_geo
H_img^c = J StopGrad(H_img)

CKA(H_geo, H_img) =
  ||(H_geo^c)^T H_img^c||_F^2
  / (
      ||(H_geo^c)^T H_geo^c||_F
      ||(H_img^c)^T H_img^c||_F
      + epsilon
    )

L_sem =
  1 - CKA(H_geo, StopGrad(H_img))
```

`L_sem` 保持 Qwen 原生视觉 token 的相对语义结构，约束几何 residual 不任意扭曲 token manifold，但不要求每个增强 token 与原 token 逐元素相同。它与 `alpha_pe=alpha_rel=0` 的初始化共同形成平滑过渡：训练初始状态严格等于原始 Qwen，随后仅在几何目标提供收益且 CKA 漂移受控时增加 residual。

### 3.12 VLM 空间推理与规划

增强后的视觉 token 与普通文本 token 一起输入 Qwen LLM：

```text
Y = Qwen_LLM_LoRA(H_geo, H_text)
```

普通场景描述和空间问答使用原始 language head。LLM 仅通过 rank-16 LoRA 微调。

对于包含精确坐标的输入或规划输出，我们采用与 SpaceDrive 对齐的可选 coordinate interface：

- 文本中的 `(x,y,z)` 坐标使用同一个 `Phi` 编码；
- 在坐标 token 前加入特殊标记 `<POS>`；
- 输出 `<POS>` 后，将对应 hidden state 送入 MLP coordinate decoder；
- 轨迹回归使用 Huber loss，而不是逐 digit 生成坐标。

该接口用于规划评测，不作为本文主要创新。

### 3.13 训练目标与训练阶段

#### 阶段 A：几何适配器预训练

冻结：

- Qwen Vision Encoder；
- Qwen merger；
- Qwen LLM。

仅训练：

- token geometry encoder；
- factorized allocation head 与 sub-token ray head；
- R²AC depth head；
- relation edge encoder 与预测 heads；
- FGT feature/structure cost adapters。

阶段 A 分为两个连续子阶段。前 `T_warm` steps 热启动可独立监督的 allocation、ray、relation heads 与 FGT cost adapters：

```text
L_A0 =
  lambda_alloc L_alloc
  + lambda_ray L_ray
  + lambda_rel L_rel
  + lambda_fgt L_FGT
```

随后加入由 transport barycenter 固定构造的 oracle 压扩目标 `zeta_p^*=F(d_p^T;a_p^*)`、metric coordinate 和 ranking 监督。完整阶段 A 目标为：

```text
L_geo =
  L_depth
  + lambda_coord L_coord
  + lambda_ray L_ray
  + lambda_alloc L_alloc
  + lambda_rel L_rel

L_A =
  L_geo
  + lambda_fgt L_FGT
```

`A0` 与完整阶段 A 使用相同的 `lambda_alloc`、`lambda_ray`、`lambda_rel` 和 `lambda_fgt`；warm-up 时令 feature cost 中的 `lambda_d=0`，避免尚未稳定的 depth prior 影响 coupling。warm-up 的首要目的是在 metric 解码和 coordinate loss 启用前，使 `r_hat_p`、`q_hat_p` 及其解析组合 `a_hat_p` 接近教师目标，并使 FGT coupling 与 relation feature 先由投影、射线和结构项稳定下来。随后启用 `L_depth`、`L_coord` 和可选的 stopped-gradient depth prior。

阶段 A 将 `alpha_pe` 和 `alpha_rel` 固定为 0，不向 LLM 注入几何。relation supervision 此时只训练 standalone edge encoder 与预测 heads，`W_up` 不参与阶段 A。

#### 阶段 B：VLM 空间指令微调

继续冻结：

- Vision Encoder；
- merger。

训练：

- geometry minibatch 上的 FGT cost adapters；
- LPGA；
- 3D PE / relation residual gates；
- relation message projection `W_up`；
- Qwen LLM LoRA；
- 可选 coordinate decoder。

目标：

```text
L_B =
  L_LM
+ lambda_plan  L_plan
+ lambda_geo   L_geo
+ lambda_fgt   L_FGT
+ lambda_keep  L_keep
+ lambda_sem   L_sem
```

阶段 B 中 `lambda_fgt` 小于阶段 A，仅在包含同步 LiDAR 的 geometry minibatch 上启用，用于防止 instruction tuning 后 coupling、depth teacher 和 relation structure 漂移；无 LiDAR 的语言 minibatch 令该项为 0。`lambda_sem` 与 `lambda_keep` 分别限制 token-level relation drift 和全局 pooled drift。

其中全局保持损失作用于 pooled 视觉表示：

```text
L_keep =
  1 - cosine(
    Pool(H_geo),
    StopGrad(Pool(H_img))
  )
```

`Pool` 先对每个相机的有效视觉 token 做 mean pooling，再对相机表征取平均，得到 `R^3584` 全局视觉表示。它限制整体语义漂移，但不要求每个增强 token 与原 token 完全相同。

#### 阶段 C：可选鲁棒训练

随机执行：

- single-camera drop；
- multi-camera drop；
- front-only；
- image occlusion；
- calibration noise；
- low-light corruption。

完整输入作为 teacher，退化输入作为 student：

```text
L_robust =
  ||Pool(H_geo^degraded)
  - StopGrad(Pool(H_geo^full))||_2
```

### 3.14 训练与推理模块状态

| 模块 | 阶段 A | 阶段 B | 推理 |
|---|---|---|---|
| Qwen Vision Encoder | 冻结 | 冻结 | 使用 |
| Qwen merger / projector | 冻结 | 冻结 | 使用 |
| LiDAR teacher / FGT solver | 构造 coupling 与 teacher | 仅 geometry minibatch 低权重使用 | 移除 |
| FGT cost adapters | 训练 | 仅 geometry minibatch 低权重训练 | 移除 |
| 外部 UniDepth | 不使用 | 不使用 | 不使用 |
| LPGA R²AC depth head | 训练 | 训练 | 使用 |
| LPGA allocation head | 训练 | 训练 | 使用 |
| LPGA sub-token ray head | 训练 | 训练 | 使用 |
| Relation edge encoder | 训练 | 训练 | 使用 |
| Relation projection `W_up` | 未接入 | 小非零初始化后训练 | 使用 |
| 3D PE / residual gates | 固定为 0 | 训练 | 使用 |
| `L_sem` / `L_keep` | 未注入，无需启用 | 训练约束 | 移除 |
| Qwen LLM | 冻结 | LoRA rank=16 | 使用 |
| Coordinate decoder | 未启用 | 可选训练 | 可选使用 |

---

## 4. 实验

### 4.1 数据集

**nuScenes [4]。** 主要训练和评测数据集，提供六相机、LiDAR、ego pose、标定、3D box 和地图信息。LiDAR 仅用于训练标签。

**OmniDrive / DriveLM 风格 QA [5,6]。** 用于场景描述、空间问答、反事实推理和规划指令微调。

**KITTI-360 [12]。** 用于跨数据集 metric depth、跨视角几何和定位评测。

**Bench2Drive [13]。** 用于闭环规划评测，报告 Driving Score、Success Rate、Route Completion 和 Infraction Score。主协议不使用 Bench2Drive-VL [14] 的额外语言标注；若增加该数据训练，则作为独立实验行报告，不与主协议混合。本文对驾驶能力的主张同时受 nuScenes 开环和 Bench2Drive 闭环结果约束。

### 4.2 实现配置

默认 Base VLM：

```text
Qwen2.5-VL-7B-Instruct
```

默认图像设置：

- 六相机输入；
- 输入分辨率 `640 x 640`，对齐 SpaceDrive 主配置；
- 动态 merged token 网格约为 `6 x 22-23 x 22-23`，实际形状由 `image_grid_thw` 决定；
- 使用 Qwen 原生 image processor；
- 保留 `image_grid_thw` 和每个相机 token 边界；
- Vision Encoder 与 merger 使用预训练权重并冻结。

LPGA 默认配置：

| 模块 | 默认值 |
|---|---:|
| Geometry bottleneck | 512 |
| Token adapter layers | 2 |
| Relation neighbors `P` | 16 或 32 |
| LiDAR anchor type | TBD |
| FGT local candidates per token | TBD |
| Feature-structure trade-off `gamma` | TBD |
| Entropic regularization `epsilon_ot` | TBD |
| Partial transport budget `m_tr` | TBD |
| Teacher reliability fusion `eta_q` | TBD |
| Partial mass threshold `tau_mass` | TBD |
| Depth representation | R²AC |
| Maximum depth `D_max` | 80 m |
| Maximum sensitivity ratio `exp(beta)` | 16 |
| Reliability support `n_0` | 3 |
| Reliability scale `tau_q` | 0.1 |
| Minimum depth weight `epsilon_q` | 0.05 |
| Allocation factor ratio `lambda_r : lambda_q` | 0.5 : 0.5，且和为 1 |
| Allocation total weight `lambda_alloc` | 1.0 |
| Foreground band `tau_fg` | 1.0 m |
| Ranking temperature `tau_d` | 1.0 m |
| Minimum risk allocation `eta` | 0.5 |
| Reaction time `t_react` | 1.0 s |
| Braking deceleration `a_brake` | 4.0 m/s² |
| Safety margin `d_margin` | 2.0 m |
| Risk temperature `tau_r` | 2.0 m |
| Front temperature `tau_front` | 1.0 m |
| Corridor width `w_corridor` | 2.0 m |
| Cross-view voxel size | 0.5 m |
| Cross-view depth tolerance `tau_depth` | 1.0 m |
| PE frequency scale `tau` | 10000 |
| PE / relation maximum gate | 1.0 / 1.0 |
| Allocation warm-up `T_warm` | Stage A 前 5% steps |
| Relation projection init `sigma_up` | 1e-4 |
| LoRA rank | 16 |
| PE gate init | 0 |
| Relation gate init | 0 |

停车距离采用常减速度运动学模型。上述参数是默认训练配置而非道路法规常数，并对 `gamma`、`epsilon_ot`、候选 anchor 数、`eta_q`、`t_react`、`a_brake`、`w_corridor`、`D_max` 和 `exp(beta)` 做敏感性分析；同时报告训练集 `d_safe`、transported mass 与 coupling entropy 分布，避免只给单点超参数。

### 4.3 主要评测任务

#### 驾驶空间问答

问题类别：

- metric distance；
- relative direction；
- object comparison；
- free-space topology；
- occlusion；
- cross-view correspondence；
- temporal motion；
- counterfactual trajectory reasoning。

报告：

- overall accuracy；
- category accuracy；
- metric tolerance accuracy；
- hard paraphrase accuracy；
- object-composition accuracy。

#### Token 几何探针

冻结 Qwen Vision Encoder、merger 和 LPGA，训练同容量线性或两层 probe：

- token depth：AbsRel、RMSE、delta1；
- 近中远距离：`0-10m`、`10-30m`、`30m+`；
- factor calibration：`r_hat_p` 对 `r_p`、`q_hat_p` 对 `q_p^teacher`、`a_hat_p` 对 `a_p^*` 的 MAE 与 Spearman correlation；
- risk-stratified depth：按 `a_p^*` 分位数报告高风险与低风险 AbsRel；
- transport concentration：`q_p^OT` 与 token depth error 的 Spearman correlation，以及按 `q_p^OT` 分桶的 cross-view accuracy；
- relative 3D displacement；
- depth ordering；
- cross-view matching；
- occlusion accuracy；
- semantic retention：`CKA(H_geo,H_img)`。

为联合衡量几何收益与语义代价，定义：

```text
GeoGain / SemanticDrift =
  (Probe(H_geo) - Probe(H_img))
  / (1 - CKA(H_geo, H_img) + epsilon)
```

其中 `Probe` 统一转换为 higher-is-better 分数；对 AbsRel、RMSE 等误差指标使用相对误差下降量。该比值只作为诊断指标，不替代原始几何与语义结果。

#### nuScenes 开环规划

预测 3 秒内 6 个 waypoint，报告：

- 1s / 2s / 3s L2；
- average L2；
- collision rate；
- intersection rate。

#### Bench2Drive 闭环规划

在相同路由、交通密度和传感器配置下报告：

- Driving Score；
- Success Rate；
- Route Completion；
- Infraction Score；
- collision / off-road / red-light frequency。

#### 鲁棒性

测试：

- 单相机缺失；
- 多相机缺失；
- front-only；
- 图像随机遮挡；
- 夜间和低照度；
- calibration perturbation。

报告：

```text
Robustness Ratio =
  Performance_corrupt / Performance_clean
```

#### 推理效率

与外部深度网络方案比较：

- 参数量；
- GPU 显存；
- 每帧 latency；
- FLOPs；
- 六相机吞吐量。

核心比较是：

```text
Qwen + UniDepth + 3D PE
vs
Qwen + LPGA
```

该比较是方法成立性的必要检验：除整体 AbsRel 外，必须同时报告 `30m+` 深度、空间 QA、规划指标、参数量、显存和六相机延迟，展示 LPGA 相对外部深度网络的精度—效率 Pareto，而不能只报告平均深度指标。

### 4.4 Baselines

1. 原始 Qwen2.5-VL-7B-Instruct。
2. Qwen2.5-VL + rank-16 LoRA。
3. SpaceDrive-style：冻结 UniDepthV2 + explicit 3D PE，不使用 LiDAR 校准。
4. LiDAR-calibrated SpaceDrive：UniDepthV2 输出使用与 LPGA 相同的 LiDAR depth/coordinate losses 训练轻量 calibration head。
5. Qwen + raw-depth adapter。
6. Qwen + log-depth adapter。
7. Qwen + 固定 power-warp depth adapter。
8. Qwen + 固定 log-companding adapter。
9. Qwen + R²AC，常数强度 `a_p = a_const`，训练与推理均固定。
10. Qwen + R²AC，仅 risk target。
11. Qwen + R²AC，仅 reliability target。
12. Qwen + R²AC，oracle target 与 oracle `a_p^*` 解码，用于估计压扩上限。
13. Qwen + relation adapter only。
14. Qwen + R²AC depth + 3D PE，无 relation residual。
15. Qwen + relation residual，无 explicit 3D PE。
16. 完整 GeoDistill-VLM（NTL-FGT + R²AC + 3D PE + relation residual）。
17. 完整方法但解冻 Qwen Vision Encoder。
18. LLaVA-1.5-7B + CLIP ViT-L/14 变体。

所有 Qwen 方法使用相同视觉输入、LoRA rank 和任务数据。Hard projection、hard assignment、entropic OT 与完整 FGT 使用相同 LiDAR sweep、anchor 数和标定预算，区别仅在 teacher 构造与目标函数；raw/log/R²AC depth adapter 使用由对应 teacher 产生的同等 depth/coordinate supervision。对冻结外部深度网络同时报告无 LiDAR 的原始版本和使用同等 LiDAR 标签的 calibration 版本。3D box、occupancy 和地图标签只用于声明启用 object/topology/occlusion structure 或 relation loss 的方法，并单独报告标签预算。

### 4.5 主结果

**表 1：驾驶空间问答。**

| 方法 | Overall | Distance | Direction | Topology | Occlusion | Cross-view | Temporal | Hard Para. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen + LoRA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SpaceDrive-style PE | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| LiDAR-calibrated SpaceDrive | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| R²AC Depth | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Relation Adapter | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| GeoDistill-VLM | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

**表 2：冻结 token 几何探针。**

| 方法 | AbsRel ↓ | 0-10m ↓ | 10-30m ↓ | 30m+ ↓ | Risk ρ ↑ | Reli. ρ ↑ | Alloc. ρ ↑ | Delta-3D ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| UniDepthV2 | TBD | TBD | TBD | TBD | N/A | N/A | N/A | TBD |
| Raw Depth | TBD | TBD | TBD | TBD | N/A | N/A | N/A | TBD |
| Fixed Power-warp | TBD | TBD | TBD | TBD | N/A | N/A | N/A | TBD |
| Fixed Log-companding | TBD | TBD | TBD | TBD | N/A | N/A | N/A | TBD |
| R²AC, `a=a_const` | TBD | TBD | TBD | TBD | N/A | N/A | N/A | TBD |
| R²AC, predicted `a` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| R²AC, oracle target & decode | TBD | TBD | TBD | TBD | N/A | N/A | Oracle | TBD |
| R²AC + Relation | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| NTL-FGT + R²AC + Relation | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

**表 3：冻结 relation probe。**

| 方法 | Relative 3D ↓ | Order ↑ | Cross-view ↑ | Occlusion ↑ | Topology ↑ |
|---|---:|---:|---:|---:|---:|
| Qwen image tokens | TBD | TBD | TBD | TBD | TBD |
| R²AC Depth | TBD | TBD | TBD | TBD | TBD |
| Relation, mean aggregation | TBD | TBD | TBD | TBD | TBD |
| Relation, confidence attention | TBD | TBD | TBD | TBD | TBD |
| GeoDistill-VLM | TBD | TBD | TBD | TBD | TBD |

**表 4：nuScenes 开环规划。**

| 方法 | L2 1s ↓ | L2 2s ↓ | L2 3s ↓ | Avg. L2 ↓ | Collision ↓ | Intersection ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Qwen + LoRA | TBD | TBD | TBD | TBD | TBD | TBD |
| SpaceDrive-style | TBD | TBD | TBD | TBD | TBD | TBD |
| LiDAR-calibrated SpaceDrive | TBD | TBD | TBD | TBD | TBD | TBD |
| GeoDistill-VLM w/o relation | TBD | TBD | TBD | TBD | TBD | TBD |
| GeoDistill-VLM | TBD | TBD | TBD | TBD | TBD | TBD |

**表 5：Bench2Drive 闭环驾驶。**

主表使用 Bench2Drive 原始训练与评测协议，不使用 Bench2Drive-VL 额外语言标注；使用后者的结果必须另列并标注额外数据。

| 方法 | Driving Score ↑ | Success Rate ↑ | Route Completion ↑ | Infraction Score ↑ | Collision ↓ |
|---|---:|---:|---:|---:|---:|
| Qwen + LoRA | TBD | TBD | TBD | TBD | TBD |
| SpaceDrive-style | TBD | TBD | TBD | TBD | TBD |
| LiDAR-calibrated SpaceDrive | TBD | TBD | TBD | TBD | TBD |
| GeoDistill-VLM w/o relation | TBD | TBD | TBD | TBD | TBD |
| GeoDistill-VLM | TBD | TBD | TBD | TBD | TBD |

**表 6：推理效率。**

| 方法 | 外部 Depth | 新增参数 | 相对 UniDepthV2-L 参数 | 显存 | Latency | 30m+ AbsRel ↓ | QA ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen | 无 | 0 | N/A | TBD | TBD | N/A | TBD |
| SpaceDrive-style | UniDepthV2-L | TBD | 100% | TBD | TBD | TBD | TBD |
| LiDAR-calibrated SpaceDrive | UniDepthV2-L + calibration head | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen + LPGA | 无 | TBD | TBD | TBD | TBD | TBD | TBD |

**表 7：NTL-FGT teacher 与语义保持消融。**

| Teacher / Transport | Feature | Structure | Entropic | `L_sem` | AbsRel ↓ | Cross-view ↑ | CKA ↑ | `rho(q^OT,error)` ↓ | GeoGain / Drift ↑ |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| Hard projection + 10% quantile | N/A | N/A | 否 | 是 | TBD | TBD | TBD | TBD | TBD |
| Hard assignment | 是 | 是 | 否 | 是 | TBD | TBD | TBD | TBD | TBD |
| Entropic OT, `gamma=0` | 是 | 否 | 是 | 是 | TBD | TBD | TBD | TBD | TBD |
| GW-only | 否 | 是 | 是 | 是 | TBD | TBD | TBD | TBD | TBD |
| Full NTL-FGT w/o `L_sem` | 是 | 是 | 是 | 否 | TBD | TBD | TBD | TBD | TBD |
| Full NTL-FGT | 是 | 是 | 是 | 是 | TBD | TBD | TBD | TBD | TBD |

### 4.6 消融实验

#### Native Token–LiDAR Fused Geometric Transport

- hard projection label vs transport barycenter teacher；
- 10% quantile depth `d_p^Q` vs FGT barycenter depth `bar d_p^L`；
- hard assignment vs entropic soft coupling；
- `gamma=0`，移除 structure term；
- 只保留 GW structure，移除 feature term；
- balanced OT vs partial/relaxed marginal；
- point、voxel 与 local surface anchor；
- 不同 `epsilon_ot`、`gamma`、`m_tr`、候选 anchor 数和 Sinkhorn 迭代数；
- `q_p^OT=q_p^conc` vs `q_p^OT=q_p^conc q_p^mass`；
- `q_p^teacher=q_p^geom`、`q_p^teacher=q_p^OT` 与二者加权融合；
- with / without `L_sem`；
- coupling 使用 predicted depth prior vs `lambda_d=0`；
- transport barycenter 对 depth、ray 和 coordinate 同时监督 vs 仅监督 depth。

该组实验同时报告 CKA、GeoGain / SemanticDrift、transported mass、coupling entropy，以及 `q_p^OT` 与 depth error / cross-view accuracy 的相关性，避免只用最终任务分数判断 transport 是否有效。

#### Vision Encoder 与注入位置

- 冻结 Qwen visual + merger；
- 解冻 Qwen visual 最后 4 层；
- 全量解冻 visual；
- 在 merger 前注入；
- 在 merger 后注入；
- 在 LLM 前增加额外 projector；
- 直接作用于原生 `image_embeds`。

该消融验证：几何收益是否必须依赖重新训练 Vision Encoder。

#### 深度表示与监督

- 无深度；
- 冻结 UniDepthV2；
- LiDAR-supervised raw depth adapter；
- log-depth adapter；
- 固定 power-warp adapter；
- 固定 log-companding adapter；
- R²AC 使用常数强度 `a=a_const`；
- 仅 `L_r`，固定 `q_hat_p=1`，形成 risk-only allocation；
- 仅 `L_q`，固定 `r_hat_p=1`，形成 reliability-only allocation；
- 仅 `L_a^aux` 直接监督组合强度，移除 `L_r` 与 `L_q`；
- 在主方法上额外加入辅助 `lambda_a L_a^aux`；
- joint allocation：单个 sigmoid head 直接预测 `a_hat_p`；
- factorized allocation：风险与几何一致性分支共享 `g_p^mono` 并解析组合（主方法）；
- factorized allocation，风险与几何一致性使用独立 token encoder；
- `L_q` 使用 SmoothL1 vs soft-target BCE；
- oracle `a_p^*` target + predicted `a_hat_p` decode；
- oracle `a_p^*` target + oracle `a_p^*` decode；
- predicted `a_hat_p` 生成移动 target；
- 无 allocation warm-up；
- R²AC 无 `StopGrad`；
- raw/log depth + 显式 `(1+kappa r_p)` loss reweighting；
- R²AC + 显式 risk reweighting；
- 不同 `D_max` 与 `exp(beta)`；
- single-sweep LiDAR；
- multi-sweep LiDAR；
- min depth；
- 10% quantile depth；
- FGT barycenter depth。

#### Token 代表射线

- 固定 token center ray；
- LiDAR 前景占用质心，仅作为 teacher；
- 预测 sub-token ray offset；
- 无 `L_ray`；
- 不同 `tau_fg`。

#### 几何注入

- 无 3D PE；
- 仅 3D PE；
- 仅 relation residual；
- 3D PE + relation residual；
- 固定 PE scale；
- learnable zero-init gate；
- 无 `L_keep`；
- 无 `L_sem`；
- CKA vs Sliced Wasserstein semantic constraint。

#### 关系监督

- 无 `L_delta`；
- 无 `L_order`；
- 无 `L_cross`；
- 无 `L_topo`；
- 无 `L_occ`；
- 同视角边；
- 跨视角边；
- hard negative；
- dense relation 与 sparse relation；
- mean aggregation；
- learned attention；
- teacher-confidence attention；
- 跨相机边错误使用 `delta_uv` 的对照实现。

#### VLM 训练

- 冻结 LLM；
- LoRA rank 8；
- LoRA rank 16；
- LoRA rank 32；
- 仅 LM loss；
- LM + geometry retention；
- digit-wise waypoint generation；
- coordinate decoder regression。

### 4.7 防捷径与因果验证

#### Image shuffle

在 batch 内打乱 `image_embeds`，保留原 LiDAR 标签和 calibration。若几何性能不下降，说明 adapter 使用了数据集或坐标先验。

#### Calibration shuffle

保持图像与 LiDAR anchor 不变，在 batch 内交换相机内外参并重新计算 feature cost。若 FGT 确实使用标定几何，其 feature transport cost、barycenter depth error 和 coupling entropy 应明显上升，`q_p^OT`、3D 坐标和跨视角性能应下降。若 coupling 与最终性能基本不变，则模型可能依赖 camera ID 或数据集位置先验。

#### Depth-label shuffle

训练或评测时打乱 LiDAR anchor 的 depth/3D coordinate，同时保持图像与 token mass 不变，验证提升是否真正来自 metric supervision。

#### Allocation-label shuffle

保持 barycenter depth 不变，分别在 batch 内打乱 `r_p`、`q_p^geom`、`q_p^OT`、`q_p^teacher` 和组合后的 `a_p^*`，评估近场深度、`30m+` 深度、碰撞率和空间问答。若性能不下降，说明 factorized allocation head 未真正利用对应监督。

#### Transport-structure shuffle

保持 feature cost 与 LiDAR 坐标不变，打乱 `C_ij^L` 中的 object/surface、visibility 和 occlusion relation，或随机置换 token edge `C_pq^V`。该实验应主要破坏 structure term、跨视角与关系 probe，而不应等价于简单删除全部 metric depth。

#### Constant-depth baseline

对所有视觉 token 使用固定深度或按相机统计得到的平均深度，再生成 3D PE。该 baseline 用于检验显式 PE 的收益是否只来自粗略视锥位置。

#### Relation-label shuffle

保留深度监督但打乱 cross-view 和 depth-order labels，验证 relation residual 的独立贡献。

### 4.8 定性分析

可视化：

1. Qwen merged token 区域、LiDAR anchor 与局部 candidate mask；
2. token-LiDAR coupling heatmap、transported mass 与 entropy；
3. hard quantile depth、transport barycenter depth 和 LPGA depth；
4. `q_p^OT` 分桶下的 depth error 与 cross-view accuracy；
5. UniDepth 与 LPGA 在夜间、遮挡和远距离的比较；
6. 预测 3D token point cloud；
7. 跨相机 token correspondence；
8. depth-order 和 occlusion edge；
9. 原始 Qwen 与增强 token 的 LLM attention 和 CKA；
10. `r_hat/q_hat/a_hat` 与 teacher target 的散点和校准曲线；
11. `d_safe`、LiDAR support 和 cross-view edge confidence 的数据分布；
12. 空间问答和规划失败案例。

跨视角标签同时报告正负边数量、可见性过滤比例、正边深度残差和按场景抽样的人工审计结果，避免仅以最终 matching accuracy 掩盖标签噪声。

建议图：

- **图 1：** GeoDistill-VLM 总体结构与训练/推理差异。
- **图 2：** Qwen Vision Encoder、merger 和几何注入位置。
- **图 3：** Native token measure、LiDAR geometric measure、soft coupling 与 transport barycenter。
- **图 4：** R²AC 的风险、transport reliability 与 token 级深度灵敏度分配。
- **图 5：** LiDAR privileged relation graph。
- **图 6：** SpaceDrive 外部深度路径与 LPGA 路径比较。
- **图 7：** 空间 QA、跨视角 grounding 和规划可视化。

---

## 5. 讨论

### 5.1 为什么直接使用 Qwen 原生 Vision Encoder

Qwen2.5-VL 的 Vision Encoder 已通过大规模预训练获得强语义和视觉理解能力。重新训练一个独立 camera encoder 会产生两套问题：

1. 新 token 与 Qwen LLM 语义空间不天然对齐，需要额外 projector 和对齐数据；
2. 几何预训练可能损失原有视觉语义能力。

GeoDistill-VLM 保留 Qwen 的完整视觉路径，只学习 residual geometry。这样既复用原生语义表示，也使训练成本集中在空间能力本身。

### 5.2 与 SpaceDrive 的关系

两者都不重新训练 Vision Encoder，并都在视觉语言 projector 之后注入空间信息。主要差异是空间来源和部署成本：

| | SpaceDrive | GeoDistill-VLM |
|---|---|---|
| Vision Encoder | 冻结 Qwen/LLaVA 原生编码器 | 冻结 Qwen/LLaVA 原生编码器 |
| 深度来源 | 推理时运行冻结 UniDepth | 训练时 LiDAR，推理时 LPGA |
| LiDAR teacher | 可选目标域校准 | NTL-FGT soft coupling 与 barycenter |
| 深度表示 | 外部模型原生输出 | FGT teacher + token 级 R²AC 连续压扩 |
| 3D PE | 外部深度直接生成 | Qwen token 预测深度后生成 |
| 关系监督 | 主要为 explicit coordinate PE | depth + cross-view + topology + occlusion |
| 推理模块 | VLM + 外部 depth estimator | VLM + 轻量 geometry adapter |
| 跨域深度先验 | 受益于大规模预训练 UniDepth，通常更强；LiDAR-calibrated 版本还同时获得目标域校准 | 依赖 LiDAR 训练域，需 KITTI-360 等跨域验证 |

因此，GeoDistill-VLM 不是重新实现 SpaceDrive 的 Vision Encoder，而是研究如何将其外部深度路径压缩为由 LiDAR 特权训练得到的原生 token 几何能力。LiDAR-calibrated SpaceDrive 同时享受预训练深度先验和 LiDAR 校准，是 OOD 精度与效率评测中的公平最强对手。

### 5.3 为什么使用 Fused Geometric Transport 而不是硬投影

硬投影或局部分位数隐含一个强假设：每个 merged token 只对应一个可由像素区域独立确定的 LiDAR 表面。该假设在物体边界、细小目标、远距离稀疏回波和跨视角重叠处容易失效。普通 feature OT 能用投影、射线和可见性产生 soft assignment，但当多个候选具有相似局部 cost 时仍可能歧义；纯 GW 又缺少相机投影这一强物理锚点。

NTL-FGT 同时保留两类约束：feature term 确保 coupling 遵守标定与局部射线几何，structure term 使被匹配的 token pair 与 LiDAR anchor pair 具有一致的相对距离、表面、可见性和遮挡关系。逐 token barycentric projection 将多个候选压缩为 metric teacher，coupling entropy 与 transported mass 则提供可用于监督分配的可靠度。该目标用于构造更有结构的 teacher，而不声称恢复唯一真实的像素级 correspondence；其有效性需要由 `gamma=0`、GW-only、hard assignment、calibration shuffle 和 transport concentration correlation 共同验证。

### 5.4 为什么使用 R²AC 而不是固定深度变换

Raw depth 对远距离绝对误差敏感，inverse/log depth 和固定 power-warp 则对所有图像位置施加相同的损失灵敏度。对于驾驶 VLM，这两类假设都过于刚性：同为 15 米的 token，位于 ego 制动走廊中的行人和侧后方静态建筑具有不同的安全价值；同一 token 区域也可能对应单一表面或前景—背景混合。

R²AC 将上述差异压缩为单个可解释变量 `a_p`，其值直接决定近端与最远端的灵敏度比 `exp(beta a_p)`。当风险低时，变换连续退化为线性深度；当风险高且 fused teacher 可靠时，变换提高近场误差在训练目标中的灵敏度。与 AdaBins/LocalBins 相比，R²AC 不预测多组离散 bin center 和概率，而是分别预测风险与融合可靠度并解析组合为单标量压扩强度，保留 metric depth 的解析逆变换，便于直接反投影到 3D PE。

固定 power-warp、固定 log-companding、常数强度 `a=a_const`、仅风险、仅可靠度、oracle target/decode 与 predicted `a_hat_p` 均作为主消融，用于区分收益来自基础函数、条件化变量还是监督信号。

### 5.5 为什么同时需要 3D PE 和 Relation Residual

3D PE 为每个视觉 token 提供显式 metric position，便于文本坐标和视觉语义直接交互。但逐 token 坐标不能完整表达：

- 两个视角中的 token 是否属于同一对象；
- 哪些区域属于同一自由空间；
- 哪个目标遮挡另一个目标；
- token 间三维结构是否在相机缺失时保持一致。

Relation residual 用于补充这些场景级结构。二者分别承担“在哪里”和“如何关联”。

### 5.6 为什么深度 head 作用于 merger 之后

Qwen merger 输出已经与 LLM hidden space 对齐，是实际进入语言模型的视觉 token。在该位置学习 depth 和 relation，有三个优点：

1. 几何监督直接作用于 VLM 真正消费的 token；
2. 不需要修改 Qwen Vision Encoder 内部结构；
3. 3D PE 可直接与 `image_embeds` 相加。

代价是 spatial merge 降低了深度分辨率。我们通过局部 partial transport、structure alignment、barycenter ray 与 sub-token offset 减少前景/背景混合，并将 hard foreground quantile 和 merger 前注入作为消融。

### 5.7 为什么推理时不使用 LiDAR、OT Solver 或 UniDepth

LiDAR 与 FGT solver 只在训练阶段提供结构化 teacher。LPGA 学习从 Qwen 视觉 token 和标准 ego state 预测部署所需的 metric geometry；relation adapter 也只使用图像特征、标定和预测坐标构边。这样部署不需要 LiDAR、coupling 求解或外部深度网络，也不增加环境感知传感器。

本文不预设轻量 LPGA 在所有像素级深度基准上超过大型 UniDepth；LPGA 的实际参数量及其相对 UniDepthV2-L 的占比由表 6 报告。核心假设是：Qwen token 已包含足够的语义尺度和局部结构，使 LPGA 能在显著更低成本下恢复 VLM 所需的 token-level metric geometry。该假设必须由表 2 和表 6 联合验证：LPGA 的 `30m+` 误差不能出现不可接受的退化，并且其空间 QA、开环/闭环规划和延迟必须形成优于外部深度路径的 Pareto。若仅降低延迟但远场深度或驾驶指标明显崩塌，则本文“替代外部深度网络”的主张不成立。

---

## 6. 局限性

GeoDistill-VLM 依赖训练阶段同步 camera-LiDAR 数据和准确标定。LiDAR 在远距离、雨雾和遮挡区域较稀疏，多帧累积虽能增加覆盖，但动态目标需要额外运动补偿。

FGT 对标定误差、candidate mask 和 LiDAR anchor 构造敏感。熵正则过强可能产生过度平滑的 barycenter，过弱则接近不稳定的 hard assignment；局部 partial coupling 与稀疏 structure sampling 降低了计算量，但仍增加训练时显存和优化复杂度。本文不主张该 coupling 是唯一真实 correspondence，必须报告 entropy、transported mass、calibration shuffle 与失败案例。

冻结 Qwen Vision Encoder 降低了训练成本，也限制了视觉底层特征适应驾驶几何的能力。若原生 token 丢失了某些细粒度边缘或小目标信息，轻量 adapter 无法完全恢复。解冻最后若干视觉层可能提高性能，但会削弱“保持原生视觉编码器”的简洁性并增加训练成本。

Token-level depth 受到 Qwen spatial merger 分辨率限制，同一 merged token 内可能包含前景和背景。Transport barycenter 与 sub-token ray offset 缓解了“深度来自前景、射线却固定在区域中心”的结构误差，但 barycentric projection 最终仍将多模态 coupling 压缩为单个代表表面，无法完整表达多层深度。

R²AC 的风险监督依赖停车距离、ego 前向走廊和少量超参数。该定义适合以车辆前向规划为主的驾驶任务，但不能完整描述侧向切入、倒车和复杂交互风险。`D_max` 与最大灵敏度比也需要跨速度、相机和数据集验证。更通用的未来方向是使用候选轨迹距离或可学习风险场替代固定前向走廊。

训练期 LiDAR teacher 与测试域之间可能存在分布偏移。LPGA 是否能泛化到不同相机内参、道路结构和天气，需要跨数据集验证。

最后，Bench2Drive 闭环分数仍不能完全证明真实道路安全；本文的结论应限定在公开数据集与仿真闭环范围内。

---

## 7. 结论

本文提出 GeoDistill-VLM，一种将训练期 LiDAR 几何蒸馏到冻结 Qwen2.5-VL 原生视觉 token 的纯视觉部署框架。核心 NTL-FGT 将 native token measure 与 LiDAR geometric measure 建模为 feature-structure fused transport，通过 soft coupling、structure alignment 和 transport barycenter 构造 token-level metric teacher。该方法不训练新的 Vision Encoder，不替换 Qwen merger，也不在推理时运行 LiDAR、OT solver 或外部 UniDepth。

训练阶段，FGT coupling 为 Qwen merged visual token 构造 barycenter depth、三维坐标、sub-token ray、transport reliability 和成对结构目标。R²AC 将可预测的风险与 fused teacher reliability 因子化为压扩强度，在保持严格单调和解析可逆的同时动态调整近远场监督灵敏度；停止梯度的 barycenter teacher、固定 oracle 压扩目标和独立 allocation warm-up 用于减少移动目标与退化解。轻量 LPGA 从原生 `image_embeds` 预测三维位置，并将 universal 3D PE 与 relation residual 通过 zero-init gate 注入视觉 token；CKA 与全局保持损失共同限制原生语义漂移。VLM 阶段仅使用 rank-16 LoRA 微调 Qwen LLM。

GeoDistill-VLM 的核心假设是：通用 VLM 的视觉语义能力可以保持冻结，缺失的三维空间信息可通过训练期 LiDAR 和轻量 residual adapter 写入其原生视觉 token。该假设最终必须由几何精度、远场稳定性、空间问答、开环/闭环驾驶和推理效率共同验证；在这些结果完整报告前，本文不主张轻量适配器普遍替代单目深度基础模型。

---

## 参考文献

[1] Qwen Team. Qwen2.5-VL Technical Report. arXiv:2502.13923, 2025.

[2] SpaceDrive: Infusing Spatial Awareness into VLM-based Autonomous Driving. arXiv:2512.10719, 2026.

[3] UniDepthV2: Universal Monocular Metric Depth Estimation Made Simpler. arXiv:2502.20110, 2025.

[4] Caesar et al. nuScenes: A Multimodal Dataset for Autonomous Driving. CVPR, 2020.

[5] DriveLM: Driving with Graph Visual Question Answering. arXiv:2312.14150, 2023.

[6] OmniDrive: A Holistic Vision-Language Dataset for Autonomous Driving with Counterfactual Reasoning. arXiv:2504.04348, 2025.

[7] SimLingo: Vision-Only Closed-Loop Autonomous Driving with Language-Action Alignment. arXiv:2503.09594, 2025.

[8] DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models. arXiv:2402.12289, 2024.

[9] Zhu et al. LLaVA-3D: A Simple yet Effective Pathway to Empowering LMMs with 3D-Awareness. arXiv:2409.18125, 2024.

[10] SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities. arXiv:2401.12168, 2024.

[11] Image Generators are Generalist Vision Learners. arXiv:2604.20329, 2026.

[12] KITTI-360: A Novel Dataset and Benchmarks for Urban Scene Understanding in 2D and 3D. TPAMI, 2022.

[13] Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-to-End Autonomous Driving. arXiv:2406.03877, 2024.

[14] Bench2Drive-VL: Benchmarks for Closed-Loop Autonomous Driving with Vision-Language Models. arXiv:2604.01259, 2026.

[15] Barron. A Power Transform. arXiv:2502.10647, 2025.

[16] Bhat et al. AdaBins: Depth Estimation Using Adaptive Bins. CVPR, 2021.

[17] Bhat et al. LocalBins: Improving Depth Estimation by Learning Local Distributions. ECCV, 2022.

[18] ITU-T Recommendation G.711: Pulse Code Modulation of Voice Frequencies.

[19] Ranftl et al. Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer. TPAMI, 2022.

[20] Birkl et al. MiDaS v3.1: A Model Zoo for Robust Monocular Relative Depth Estimation. arXiv:2307.14460, 2023.

[21] Bhat et al. ZoeDepth: Zero-Shot Transfer by Combining Relative and Metric Depth. arXiv:2302.12288, 2023.

[22] Ke et al. Marigold: Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation. CVPR, 2024.

[23] Yang et al. Depth Anything V2. arXiv:2406.09414, 2024.

[24] Hu et al. Metric3D v2: A Versatile Monocular Geometric Foundation Model for Zero-Shot Metric Depth and Surface Normal Estimation. arXiv:2404.15506, 2024.

[25] Piccinelli et al. UniDepth: Universal Monocular Metric Depth Estimation. CVPR, 2024.

[26] Li et al. BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers. ECCV, 2022.

[27] Liu et al. PETR: Position Embedding Transformation for Multi-View 3D Object Detection. ECCV, 2022.

[28] Hu et al. Planning-Oriented Autonomous Driving. CVPR, 2023.

[29] Jiang et al. VAD: Vectorized Scene Representation for Efficient Autonomous Driving. ICCV, 2023.

[30] Hinton et al. Distilling the Knowledge in a Neural Network. arXiv:1503.02531, 2015.

[31] Zhou et al. UniDistill: A Universal Cross-Modality Knowledge Distillation Framework for 3D Object Detection in Bird's-Eye View. CVPR, 2023.

[32] Hong et al. Cross-Modality Knowledge Distillation Network for Monocular 3D Object Detection. ECCV, 2022.

[33] Lapin et al. Learning Using Privileged Information: SVM+ and Weighted SVM. Neural Networks, 2014.

[34] Provodin et al. Rethinking Knowledge Transfer in Learning Using Privileged Information. arXiv:2408.14319, 2024.

[35] Shalev-Shwartz et al. On a Formal Model of Safe and Scalable Self-Driving Cars. arXiv:1708.06374, 2017.

[36] Cuturi. Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances. NeurIPS, 2013.

[37] Peyré, Cuturi, and Solomon. Gromov-Wasserstein Averaging of Kernel and Distance Matrices. ICML, 2016.

[38] Vayer et al. Optimal Transport for Structured Data with Application on Graphs. ICML, 2019.

[39] Kornblith et al. Similarity of Neural Network Representations Revisited. ICML, 2019.
