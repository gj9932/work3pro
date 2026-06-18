# GeoToken：面向自动驾驶空间推理的深度扎根 LiDAR 特权关系预训练

匿名 CVPR 投稿

Paper ID：TBD

---

## 摘要

纯相机自动驾驶视觉语言模型能够识别交通参与者并描述道路场景，但在度量与关系空间推理上仍不可靠。模型可能知道一辆车位于前方，却无法稳定判断其距离是 10 米还是 30 米，也难以回答哪辆车更近、左前方是否存在可通行空间、远处目标是否被前车遮挡，以及同一目标是否跨相邻相机可见。本文研究一个问题：能否仅在训练阶段使用 LiDAR，为部署时的纯相机模型提供可验证的三维感知能力。

我们提出 **GeoToken**，一种面向语言可访问相机 token 的深度扎根 LiDAR 特权关系预训练框架。GeoToken 不在推理阶段使用 LiDAR，也不需要微调整个大型视觉语言模型。训练时，我们首先将 LiDAR 投影到多视角相机，使用近场敏感、连续且可逆的度量深度变换，对相机 patch 或 camera ray 特征进行稀疏深度扎根；随后，将同步 LiDAR、ego pose 与标定信息转换为 ego-centric BEV 占用、拓扑和遮挡关系，并通过 **Privileged Spatial Relation Distillation（PSRD）** 蒸馏到相机 token 交互中。深度监督使视觉特征获得真实度量尺度，关系监督进一步组织自由空间、占用连通性、跨视角一致性与遮挡顺序。

固定 BEV anchor 的坐标本身即可决定 anchor 间距离和方向，因此直接预测这些标签可能产生坐标捷径。为避免模型绕过图像，我们区分坐标确定关系与观测依赖关系，并使用 visual-conditioned residual、图像置零、跨样本图像打乱和坐标移除等测试验证模型确实依赖视觉证据。最后，我们提出 **Language-Grounded Spatial Slots（LGSS）**，将空间 slot 与弱语义空间短语对齐，使冻结 VLM 能够访问预训练几何。

我们通过驾驶空间问答、冻结空间探针、坐标捷径测试、相机退化鲁棒性和定位保持评估 GeoToken。本文的核心主张是：连续度量深度负责将真实尺度扎根到相机特征中，LiDAR 特权关系负责将这些局部几何组织为可用于语言推理的场景结构，从而在保持纯相机部署的同时提升距离、方向、自由空间、遮挡、跨视角和时序推理能力。

---

## 1. 引言

视觉语言模型正在成为自动驾驶感知、问答和决策之间的重要接口。它们可以描述道路场景、识别车辆与行人、回答驾驶问题，并将视觉输入连接到规划相关语言。然而，语义识别并不等价于可靠的空间理解。纯相机模型即使能够正确识别目标，也可能无法稳定估计目标距离、判断相对方位、理解自由空间、推断遮挡顺序，或建立跨相机的一致三维对应。

这一缺陷对自动驾驶尤为重要，因为许多驾驶问题本质上是几何和关系问题：

- 哪辆车距离 ego 更近？
- 行人位于前方、左前方还是右侧？
- 左侧是否存在连续可通行空间？
- 远处车辆是否被前车遮挡？
- 前视与左前视相机中的目标是否为同一对象？
- 目标正在接近还是远离 ego？

LiDAR 能够提供准确的三维结构，但在部署时要求 LiDAR 会增加硬件成本与系统复杂度。另一方面，许多训练数据采集平台能够提供同步 LiDAR，即使最终模型只部署相机。这形成了一个自然的 privileged learning 场景：训练时使用 LiDAR 作为几何教师，推理时仅保留相机。

现有 LiDAR-to-camera 方法通常蒸馏深度图、BEV 特征、点特征、检测 logits 或全局描述子。这些目标能够改善三维感知，但与 VLM 的空间失败并不完全一致。VLM 通常需要回答对象、区域、视角和时间之间的关系，而不是只输出单个深度值或检测框。因此，仅做深度回归可能缺少关系结构，仅做关系监督又可能缺少真实的视觉度量扎根。

一个尤其容易被忽略的问题是坐标捷径。若方法在固定 BEV 网格上定义 anchor token，则任意两个 anchor 之间的距离与方向均可由坐标直接计算。当 anchor query 包含 anchor ID 或坐标位置编码时，模型可能无需读取图像，就能在距离和方向任务上取得高准确率。此时，良好的 relation probe 并不能证明相机模型获得了三维感知能力。

GeoToken 从这一问题出发，将度量扎根和关系组织分为两个互补阶段：

1. **LiDAR 特权深度扎根。** 将 LiDAR 投影到相机平面，对有效 patch 或 camera ray 提供连续 metric depth 监督，使相机特征学习真实尺度。
2. **LiDAR 特权关系蒸馏。** 将 BEV 占用、自由空间连通性、可见性和遮挡顺序蒸馏到相机 token 交互中，使局部几何形成场景级关系结构。

我们不将固定 anchor 间的距离和方向视为主要 LiDAR 知识，因为它们由坐标确定。相反，固定坐标仅作为空间查询条件；真正的 privileged supervision 来自随场景变化的 LiDAR 观测，包括 ray depth、occupancy、topology、visibility 和 occlusion。对于距离与方向推理，我们重点使用目标中心、可见表面点、ray hit point 或由视觉证据预测的占用位置，而不是仅预测固定 anchor 中心之间的几何。

为了将空间表示连接到语言，我们进一步提出 LGSS。LGSS 使用轻量 slot reader 从 BEV token 中提取对象、区域、遮挡和运动 slot，并与弱空间短语对齐。训练完成后，相机 tokenizer 与 VLM 均被冻结，仅训练小型 adapter，从而隔离预训练 token 本身的贡献。

本文贡献如下：

1. 提出一种面向自动驾驶 VLM 的深度扎根 LiDAR 特权关系预训练框架，在训练时使用 LiDAR，在推理时保持纯相机输入。
2. 提出近场敏感的连续 metric-depth grounding，将真实度量尺度直接注入相机 patch 或 ray 特征，并与场景关系学习互补。
3. 提出防坐标捷径的 PSRD，将监督重点从固定 anchor 几何转向 occupancy、topology、visibility、occlusion 和观测依赖距离关系。
4. 建立包含图像置零、跨样本打乱、坐标移除、深度分段评测和冻结 VLM 问答的验证协议，检验模型是否真正从视觉中获得三维能力。

---

## 2. 相关工作

### 2.1 自动驾驶视觉语言模型

自动驾驶 VLM 将多视角视觉感知与语言问答、场景描述和规划推理连接起来。DriveLM 将驾驶场景建模为图视觉问答，近期纯视觉闭环驾驶方法进一步探索了语言与动作之间的对齐。这些工作展示了 VLM 在驾驶任务中的潜力，但也暴露出语义能力与度量空间可靠性之间的差距。GeoToken 不替代现有 VLM，而是为冻结 VLM 提供经过空间预训练的相机 token。

### 2.2 空间 VLM 与三维视觉基础模型

现有空间 VLM 通过深度监督、三维重建、合成空间问答、坐标输入或 instruction tuning 提升空间推理能力。近期工作还表明，大规模图像生成模型能够通过可逆视觉输出学习 metric depth 和 surface normal，说明通用视觉预训练中包含强几何先验。GeoToken 借鉴其连续、可逆和近场敏感的深度表示，但不要求生成 RGB 深度图，而是将该变换用于稀疏 LiDAR metric-depth supervision。

### 2.3 LiDAR-to-Camera 蒸馏

LiDAR-to-camera 蒸馏广泛用于纯相机三维检测、BEV 感知和深度估计。已有方法通常对齐深度、BEV 特征、点特征、检测输出或全局描述子。GeoToken 与这些方法的区别在于：深度只负责度量扎根，核心关系监督面向 VLM 所需的自由空间、拓扑、遮挡、跨视角和时序推理，并显式控制固定坐标产生的捷径。

### 2.4 鲁棒纯相机驾驶

纯相机系统容易受到相机缺失、视场受限、图像遮挡、夜间、雨天和运动模糊影响。GeoToken 将鲁棒性视为核心评测维度：若预训练 token 学到的是场景几何结构，而不是局部纹理或固定坐标映射，其空间能力应在部分相机退化时表现出更好的保持率。

---

## 3. 方法

### 3.1 问题定义

训练阶段每个样本包含多视角相机序列、同步 LiDAR、ego pose 和相机标定：

```text
X_cam   = {I_t,n},  t = 1..T, n = 1..N
X_lidar = {P_t},    t = 1..T
K       = {K_n}
T_cam   = {T_ego->cam,n}
```

目标是学习纯相机 tokenizer：

```text
Z = F_theta(X_cam, K, T_cam),  Z in R^{L x C}
```

其中 `Z` 为 ego-centric BEV anchor token。LiDAR 只用于生成训练标签，不进入推理路径。推理时，tokenizer 仅接收多视角 RGB、相机标定以及可选的相邻帧。

GeoToken 的总训练目标由四部分组成：

```text
L_total =
  lambda_depth L_dg
+ lambda_psrd  L_psrd
+ lambda_lang  L_lgss
+ lambda_keep  L_keep
```

其中：

- `L_dg` 将 metric scale 扎根到相机 patch 或 ray feature；
- `L_psrd` 学习 occupancy、topology、visibility 和 occlusion 等场景关系；
- `L_lgss` 使空间表示可被语言查询；
- `L_keep` 保持原始视觉语义和定位能力。

第一阶段仅训练 `L_dg + L_psrd + L_keep`。只有冻结空间探针和捷径测试通过后，才进入 LGSS 与 VLM adapter 阶段。

### 3.2 相机空间 Tokenizer

相机 student 包含五个组件：

1. **图像编码器。** 使用预训练 DINOv2、SigLIP、CLIP-ViT 或轻量 backbone 提取 patch token。
2. **视角感知融合。** 注入 camera ID、内外参和时间编码，融合多相机特征。
3. **深度扎根头。** 从 patch 或 ray feature 预测 warped metric depth。
4. **标定感知 BEV lifting。** BEV anchor query 根据标定从可见图像区域读取特征。
5. **时序编码器。** 使用 ego-motion 对齐相邻帧后聚合时序信息。

对第 `t` 帧第 `n` 个相机：

```text
H_t,n = E_img(I_t,n)
```

注入视角和标定信息后得到：

```text
M = E_temp(E_view({H_t,n}, K, T_cam))
```

固定 BEV anchor query 通过 calibration-aware cross-attention 读取多视角证据：

```text
r_i = CrossAttn(q_i, M_visible(i))
z_i = q_i + r_i
```

其中 `q_i` 包含 anchor 坐标条件，`M_visible(i)` 为 anchor 在相机视锥中对应的可见特征，`r_i` 是 cross-attention 产生的视觉更新。PSRD 的主要观测依赖 heads 使用 `r_i` 或 `[r_i, z_i]`，而不是只使用 `z_i`。这样可以降低 relation head 直接从 anchor query 解码坐标的可能性。

模型输出：

```text
Z       = {z_i}       BEV anchor tokens
R       = {r_i}       visual-conditioned residuals
D_patch = {u_hat_p}   warped metric depth predictions
S       = {s_k}       language-grounded spatial slots
g       = Pool(Z)     global descriptor
```

### 3.3 LiDAR 特权连续深度扎根

#### LiDAR 到相机投影

训练时，将 LiDAR 点转换到中心帧 ego 坐标，再投影到每个相机：

```text
p_cam = T_ego->cam p_ego
[u, v, 1]^T ~ K p_cam
```

仅保留位于相机前方、图像范围内且通过基本可见性过滤的点。若多个 LiDAR 点落入同一 patch 或 ray bin，使用最近有效表面深度，或使用稳健分位数聚合。由此得到稀疏 metric-depth 标签：

```text
D_lidar = {(p, d_p, m_p)}
```

其中 `p` 是 patch 或 ray 索引，`d_p` 是相机坐标系中的 metric depth，`m_p` 是可靠性 mask。

#### 近场敏感的连续深度变换

直接回归原始米制深度会使远距离大数值主导损失，而驾驶任务通常更关注近场精度。我们采用连续、单调且可逆的 power transform：

```text
u = f(d; lambda, c)
  = 1 - (1 - d / (lambda c))^(lambda + 1)
```

其中 `d >= 0`，默认使用：

```text
lambda = -3
c      = 10 / 3
```

该变换将 `[0, +infinity)` 映射到 `[0, 1)`，并为近场距离分配更高分辨率。其逆变换为：

```text
d = lambda c [1 - (1 - u)^(1 / (lambda + 1))]
```

与生成式深度方法不同，GeoToken 不需要将 `u` 编码成 RGB 色图。深度头直接预测标量 `u_hat_p`：

```text
u_hat_p = h_depth(H_p)
```

深度损失为：

```text
L_depth =
  sum_p m_p SmoothL1(u_hat_p, f(d_p))
  -----------------------------------
          sum_p m_p + epsilon
```

可选地，在有效相邻 patch 之间增加局部排序损失。令
`s_pq=sign(d_q-d_p)`，则：

```text
L_rank =
  sum_pq m_pq log(
    1 + exp(-s_pq (u_hat_q - u_hat_p))
  )
```

最终深度扎根目标为：

```text
L_dg = L_depth + lambda_rank L_rank
```

默认主方法使用稀疏 LiDAR depth，不依赖额外稠密深度教师。多帧 LiDAR accumulation、box surface completion 和外部 monocular depth teacher 仅作为增强实验。

### 3.4 Ego-Centric LiDAR 场景图

我们在 ego 坐标系中定义固定 BEV 范围：

```text
x in [-40m, 40m]
y in [-40m, 40m]
cell size = 2m x 2m
```

由此得到 `40 x 40 = 1600` 个候选 anchor。一个直接但有风险的方案，是在训练时优先选择：

1. 3D box 覆盖区域；
2. 有 LiDAR 占用证据的区域；
3. 可行驶或自由空间边界；
4. 位于相机视锥内的区域；
5. 均匀采样的自由空间区域。

然而，这会产生 **selection leakage**：active anchor 的 ID 集合本身可能暴露 occupied、free-space boundary 或 object location。模型即使不读取图像，也可能利用“哪些 anchor 被选中”推断场景标签。

因此，默认主方法采用训练与推理一致的 active anchor 集：

- 根据 camera frustum visibility 选择可见 anchor；
- 在可见区域中使用固定或随机均匀采样补足 `L=400`；
- 或运行完整 1600 anchor，仅对有效 LiDAR 标签采样 loss edge；
- camera-only learned selector 只有在独立训练并通过 selection-only baseline 后才启用。

LiDAR、3D box 和 map label 只决定 supervision mask、edge 类型和 loss 权重，不决定 student 模型能够看到哪些 anchor token。LiDAR-prioritized active selection 仅作为对照实验，不作为默认设置。

#### 坐标确定关系与观测依赖关系

对固定 anchor `i,j`，以下量由坐标直接确定：

```text
d_coord_ij = ||c_i - c_j||_2
a_coord_ij = bearing(c_j - c_i)
```

它们不是 LiDAR 特权知识，主要用于：

- 构造 sparse edge neighborhood；
- 提供相对位置条件；
- 定义图上的局部与长程采样。

真正随场景变化的 LiDAR 标签包括：

```text
y_i^occ       occupancy / free / unknown
y_ij^topo     free-space or occupied connectivity
y_ij^vis      co-visibility / cross-view visibility
y_ij^occOrder occlusion order
d_i^surface   visible surface or ray-hit metric depth
d_ab^object   object/surface relation distance
```

因此主场景图定义为：

```text
G_lidar = (V, E_scene)
```

每条观测依赖边保存：

```text
e_ij = (
  y_ij^topo,
  y_ij^vis,
  y_ij^occOrder,
  d_ij^obs,
  m_ij,
  m_ij^occ
)
```

其中 `d_ij^obs` 来自 LiDAR hit、对象中心、对象表面或有效占用位置，而不是固定 anchor 中心距离。

#### 可靠性 mask

仅在标签可靠时启用监督。以下情况设置 `m_ij=0`：

- anchor 位于所有相机视场外；
- LiDAR 点数不足；
- 投影存在明显歧义；
- 动态对象跨帧关联不一致；
- free/occupied 状态不确定；
- 关系完全由坐标决定且没有视觉观测标签。

遮挡顺序只在两个目标投影到同一或相邻 ray neighborhood 且深度可分时有效，否则设置 `m_ij^occ=0`。

### 3.5 防坐标捷径的特权空间关系蒸馏

#### 观测关系分布

传统 token affinity 可写为：

```text
R_cam_ij =
  softmax_j((W_q z_i)^T (W_k z_j) / sqrt(C))
```

为降低坐标 query 的直接影响，我们使用 visual-conditioned residual 构造主要关系分布：

```text
R_cam_ij^vis =
  softmax_j((W_q r_i)^T (W_k r_j) / sqrt(C))
```

LiDAR teacher affinity 不再仅依赖固定 anchor 距离，而是由观测占用和拓扑构成：

```text
A_lidar_ij =
  exp(-alpha d_coord_ij)
  * rho(y_ij^topo)
  * kappa(y_ij^vis)
  * 1[m_ij = 1]
```

```text
R_lidar_ij =
  A_lidar_ij / sum_k A_lidar_ik
```

其中距离仅作为局部性先验，`rho` 和 `kappa` 来自场景观测。若 topology 或 visibility 未知，该 pair 不参与 relation distribution loss，而不是仅依靠坐标产生 teacher affinity。

```text
L_rel =
  sum_i KL(
    R_lidar_i,N(i)
    ||
    R_cam_i,N(i)^vis
  )
```

#### 节点与边预测

节点级 heads：

```text
y_hat_i^occ, u_hat_i^surface = phi_node(r_i)
```

边级 heads：

```text
y_hat_ij^topo,
y_hat_ij^vis,
y_hat_ij^occOrder,
u_hat_ij^obs
  = phi_edge([
      r_i,
      r_j,
      r_i - r_j,
      r_i * r_j,
      delta_c_ij
    ])
```

其中 `delta_c_ij` 作为显式几何条件输入，而不是需要从 token 中“发现”的 privileged label。

损失为：

```text
L_occNode = CE(y_hat_i^occ, y_i^occ)
L_surface = SmoothL1(u_hat_i^surface, f(d_i^surface))
L_topo    = CE(y_hat_ij^topo, y_ij^topo)
L_vis     = CE(y_hat_ij^vis, y_ij^vis)
L_occOrd  = CE(y_hat_ij^occOrder, y_ij^occOrder)
L_obsDist = SmoothL1(u_hat_ij^obs, f(d_ij^obs))
```

PSRD 总目标为：

```text
L_psrd =
  lambda_rel     L_rel
+ lambda_node    L_occNode
+ lambda_surface L_surface
+ lambda_topo    L_topo
+ lambda_vis     L_vis
+ lambda_occOrd  L_occOrd
+ lambda_obsDist L_obsDist
```

第一版默认启用：

```text
L_rel + L_occNode + L_surface + L_topo
```

`L_vis` 和 `L_occOrd` 只有在 projection/ray-neighborhood 可视化验证稳定后启用。

#### Sparse edge sampling

对每个 active anchor 最多采样 `P=32` 条边：

- `P_near`：坐标上接近且有有效观测的 anchor；
- `P_topo`：同一自由空间或占用连通分量；
- `P_occ`：有效遮挡顺序 pair；
- `P_cross`：跨相机共同可见 pair；
- `P_hard`：图像外观相似但拓扑不同的 hard negative；
- `P_rand`：随机长距离有效 pair。

在 `L=400, P=32` 时，每帧最多约 `12.8k` 条边。

### 3.6 坐标捷径控制

仅修改 loss 不能充分证明模型没有使用坐标捷径，因此我们在训练和评测中加入以下控制。

#### Query residual

主要观测依赖 heads 使用 cross-attention 的视觉更新 `r_i`，完整 token 定义为 `z_i=q_i+r_i`。坐标只作为查询条件或显式关系条件，不允许将固定 anchor 距离预测包装为 LiDAR 蒸馏收益。

#### Image-zero test

将所有图像置零或替换为均值图像，保留 anchor ID、坐标和标定：

```text
Z_zero = F_theta(0, K, T_cam)
```

如果 occupancy、surface depth、topology 或 occlusion 性能仍接近正常图像，说明模型主要使用坐标先验。

#### Cross-sample shuffle

在 batch 内打乱图像，但保留原样本 LiDAR 标签和 anchor：

```text
Z_shuffle = F_theta(X_cam[perm], K, T_cam)
```

观测依赖任务应显著下降，而纯坐标任务可保持不变。

#### Coordinate ablation

移除 anchor ID embedding、coordinate projection 或显式 `delta_c_ij`，分别评估视觉证据与坐标条件的贡献。

#### Coordinate-only baseline

训练只接收 anchor 坐标的 MLP/Transformer baseline。主方法必须在 occupancy、surface depth、topology、occlusion 和空间 QA 上显著超过该 baseline。

#### Selection-only baseline

模型只接收 active anchor ID/mask，不读取 RGB、LiDAR feature 或完整坐标特征。若该 baseline 能预测 occupancy 或 object region，说明 active selection 泄漏了标签。默认主方法要求 selection-only 结果接近类别先验。

### 3.7 Language-Grounded Spatial Slots

PSRD 使相机 token 具备空间结构，但冻结 VLM 不一定能够直接查询这些结构。LGSS 使用 `K` 个可学习 slot query：

```text
s_k = Attn(q_k^slot, Z)
```

slot 分为：

- **对象 slot**：读取动态目标或 3D box 覆盖区域；
- **区域 slot**：读取自由空间、占用区域和道路边界；
- **遮挡 slot**：读取共享 ray neighborhood 中的前后顺序；
- **跨视角 slot**：读取相邻相机的共同可见区域；
- **运动 slot**：读取 ego-motion 对齐后的时序变化。

使用 LiDAR、3D box、地图和 metadata 生成弱空间短语，例如：

- “前方约 10 米处的车辆”
- “左前方连续可通行区域”
- “被前车遮挡的远处车辆”
- “同时出现在前视和左前视相机中的车辆”
- “正在接近 ego 的行人”

LGSS 使用多正样本对比学习：

```text
L_phrase =
  -log
  exp(sim(s_k, t_j) / tau)
  -------------------------
  sum_l exp(sim(s_k, t_l) / tau)
```

并使用 slot attention 与伪目标区域之间的 grounding loss：

```text
L_ground = 1 - IoU(A_k, A_j^target)
```

```text
L_lgss =
  lambda_phrase L_phrase
+ lambda_ground L_ground
+ lambda_div     L_diversity
```

LGSS 不负责学习基础几何，其作用是将深度扎根和 PSRD 学到的空间结构暴露给语言接口。

### 3.8 表征保持与传感器退化一致性

为避免几何预训练破坏原有视觉语义和定位能力，我们使用冻结 backbone 特征或原始全局描述子作为 teacher：

```text
L_keep = 1 - cosine(g, StopGrad(g_base))
```

训练时随机执行单相机缺失、多相机缺失、视场裁剪和图像遮挡。退化输入与完整输入的空间表示保持一致：

```text
L_deg =
  ||Pool(Z_degraded) - StopGrad(Pool(Z_full))||_2
```

该策略与低比例视觉任务混训的思想一致：加入几何目标时，不应牺牲预训练模型原有的通用视觉能力。

### 3.9 冻结 VLM Adapter

完成几何预训练和 LGSS 后，冻结：

- 图像 backbone；
- camera tokenizer；
- 深度 head；
- LGSS；
- 目标 VLM。

仅训练轻量 adapter：

```text
T_adapter = A_psi([S; Pool(Z)])
```

adapter 可使用 MLP projector 或轻量 Q-Former。所有 baseline 使用相同 VLM、相同 adapter 容量和相同 QA 数据，从而将性能差异归因于视觉 token 的空间质量。

---

## 4. 实验

### 4.1 数据集

**nuScenes。** 主要训练与评测数据集，提供六视角相机、LiDAR、ego pose、标定、3D box 和场景 metadata。LiDAR 仅用于预训练标签生成，所有最终推理均只使用 RGB。

**DriveLM-nuScenes 与空间 QA。** 使用 DriveLM 风格标注，并基于 3D box、LiDAR relation graph 和 ego pose 构造距离、方向、自由空间、遮挡、跨视角和时序问题。

**KITTI-360。** 用于跨数据集深度、几何与定位探针。由于相机配置与 nuScenes 不同，将其作为 OOD 测试。

### 4.2 评测任务

#### 驾驶空间 QA

问题分为：

- 距离：哪个目标更近，目标位于哪个距离范围；
- 方向：目标相对 ego 的方位；
- 自由空间：前方或侧方是否存在连续可通行区域；
- 遮挡：哪个目标位于前方，哪个目标被遮挡；
- 跨视角：相邻相机中是否为同一目标；
- 时序：目标正在接近还是远离。

报告 overall accuracy、category accuracy、hard paraphrase accuracy 和 object-composition accuracy。

#### 冻结空间探针

冻结 camera tokenizer，训练同容量小型 head：

- sparse/dense depth：AbsRel、RMSE、delta1；
- 近场深度：`0-10m`、`10-30m`、`30m+` 分段指标；
- BEV occupancy：IoU、mIoU；
- topology、visibility、occlusion accuracy；
- object/surface distance accuracy；
- VPR Recall@1/5/10 和 pose error。

固定 anchor 中心距离和方向不作为证明三维感知能力的主要 probe，因为 coordinate-only baseline 可直接求解。

#### 坐标捷径测试

报告：

```text
Visual Dependency Drop =
  Performance_normal - Performance_zero/shuffle
```

对比：

- 正常图像；
- 图像置零；
- batch 内图像打乱；
- coordinate-only；
- selection-only；
- 无 anchor ID embedding；
- 无 coordinate projection；
- query token 与 visual residual 两种 head 输入。

观测依赖任务在 zero/shuffle 下应明显下降；若不下降，则该指标不能作为相机三维感知证据。

#### 鲁棒性

评估：

- single camera drop；
- multiple camera drop；
- front-only；
- limited FOV；
- random image occlusion；
- night；
- rain；
- motion blur。

报告：

```text
Robustness Ratio =
  Performance_corrupt / Performance_clean
```

#### 定位与语义保持

使用 VPR 和原始视觉任务验证几何训练没有破坏 backbone 能力。报告 Recall@1/5/10、max F1，以及可选的线性分类或语义检索保持率。

### 4.3 Baselines

1. 冻结 VLM 原始视觉输入。
2. CLIP/SigLIP patch token + 同容量 adapter。
3. DINOv2 patch token + 同容量 adapter。
4. DINOv2 + temporal encoder。
5. DINOv2 + raw metric-depth regression。
6. DINOv2 + log-depth regression。
7. DINOv2 + warped continuous depth grounding。
8. DINOv2 + BEV occupancy auxiliary。
9. Descriptor-only LiDAR distillation。
10. Coordinate-only BEV anchor model。
11. Selection-only active anchor model。
12. 原始 PSRD：固定 anchor distance/direction/topology。
13. 防捷径 PSRD，不使用 depth grounding。
14. Depth grounding + 防捷径 PSRD，不使用 LGSS。
15. 完整 GeoToken。

### 4.4 主结果

**表 1：nuScenes 驾驶空间 QA。**

| 方法 | Overall | Distance | Direction | Free Space | Occlusion | Cross-view | Temporal | Hard Para. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen VLM | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DINOv2 Adapter | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Depth Grounding | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Original PSRD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Shortcut-Resistant PSRD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| GeoToken w/o LGSS | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| GeoToken | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

**表 2：冻结空间探针。**

| 方法 | Depth AbsRel ↓ | 0-10m AbsRel ↓ | BEV IoU ↑ | Topology ↑ | Occlusion ↑ | VPR R@1 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| DINOv2 | TBD | TBD | TBD | TBD | TBD | TBD |
| Raw Depth | TBD | TBD | TBD | TBD | TBD | TBD |
| Log Depth | TBD | TBD | TBD | TBD | TBD | TBD |
| Warped Depth | TBD | TBD | TBD | TBD | TBD | TBD |
| PSRD | TBD | TBD | TBD | TBD | TBD | TBD |
| Warped Depth + PSRD | TBD | TBD | TBD | TBD | TBD | TBD |

**表 3：坐标捷径测试。**

| 方法 | 输入 | Surface Depth ↑ | Occupancy ↑ | Topology ↑ | Occlusion ↑ | Spatial QA ↑ |
|---|---|---:|---:|---:|---:|---:|
| Coordinate-only | Coordinates | TBD | TBD | TBD | TBD | TBD |
| Selection-only | Active IDs/Mask | TBD | TBD | TBD | TBD | TBD |
| Original PSRD | Normal RGB | TBD | TBD | TBD | TBD | TBD |
| Original PSRD | Zero RGB | TBD | TBD | TBD | TBD | TBD |
| Original PSRD | Shuffled RGB | TBD | TBD | TBD | TBD | TBD |
| GeoToken | Normal RGB | TBD | TBD | TBD | TBD | TBD |
| GeoToken | Zero RGB | TBD | TBD | TBD | TBD | TBD |
| GeoToken | Shuffled RGB | TBD | TBD | TBD | TBD | TBD |

**表 4：相机退化鲁棒性。**

| 方法 | Clean QA | Robust QA | QA Ratio | Clean VPR | Robust VPR | VPR Ratio |
|---|---:|---:|---:|---:|---:|---:|
| DINOv2 | TBD | TBD | TBD | TBD | TBD | TBD |
| DINOv2 + Dropout | TBD | TBD | TBD | TBD | TBD | TBD |
| Warped Depth | TBD | TBD | TBD | TBD | TBD | TBD |
| PSRD | TBD | TBD | TBD | TBD | TBD | TBD |
| GeoToken | TBD | TBD | TBD | TBD | TBD | TBD |

### 4.5 消融实验

#### 深度表示

- 不使用 depth grounding；
- raw metric depth；
- inverse depth；
- log depth；
- depth bin classification；
- power-warp continuous depth；
- power-warp depth + local ranking；
- 不同 `lambda` 与 `c`；
- sparse single-frame LiDAR；
- multi-frame LiDAR accumulation；
- external dense depth teacher。

#### PSRD 标签

- 固定 anchor distance/direction；
- 移除固定 anchor distance/direction supervision；
- 无 occupancy；
- 无 topology；
- 无 visibility；
- 无 occlusion order；
- 无 observed surface/object distance；
- 坐标距离只作为 edge prior；
- 坐标距离同时作为预测目标。

#### 防捷径设计

- relation head 使用完整 `z_i`；
- relation head 使用 residual `r_i`；
- 不使用 anchor ID embedding；
- 不使用 coordinate projection；
- 不使用显式 `delta_c_ij`；
- coordinate-only baseline；
- selection-only baseline；
- normal/zero/shuffle 输入。

#### LGSS

- 无 LGSS；
- shuffled phrase；
- CLIP text space；
- mismatched text encoder；
- target-VLM text space；
- hard paraphrase split；
- object-composition split。

#### 输入与架构

| 设置 | Spatial QA | Robust QA | VPR R@1 |
|---|---:|---:|---:|
| T=1 | TBD | TBD | TBD |
| T=3 | TBD | TBD | TBD |
| T=5 | TBD | TBD | TBD |
| Front-only | TBD | TBD | TBD |
| 3 cameras | TBD | TBD | TBD |
| 6 cameras | TBD | TBD | TBD |
| 400 anchors | TBD | TBD | TBD |
| 1600 anchors | TBD | TBD | TBD |
| MLP adapter | TBD | TBD | TBD |
| Q-Former adapter | TBD | TBD | TBD |

### 4.6 定性分析

可视化包括：

1. LiDAR 点投影与 sparse metric-depth 标签；
2. 原始深度、log-depth 和 power-warp depth 的分辨率分配；
3. BEV occupancy/free/unknown map；
4. active anchor 与多相机投影；
5. topology、visibility 和 occlusion edge；
6. 正常、图像置零和图像打乱条件下的 token affinity；
7. LGSS attention 与语言短语；
8. 相机缺失、夜间、稀疏 LiDAR 和重遮挡失败案例。

建议图：

- **图 1：** GeoToken 总体框架。LiDAR depth grounding → camera BEV lifting → shortcut-resistant PSRD → LGSS → frozen VLM。
- **图 2：** 近场敏感连续深度变换及其逆变换。
- **图 3：** 坐标确定关系与观测依赖关系的区别。
- **图 4：** LiDAR ego-centric scene graph 构建与 selection leakage 控制。
- **图 5：** normal/zero/shuffle 坐标捷径测试。
- **图 6：** LGSS 与冻结 VLM adapter。

---

## 5. 讨论

### 5.1 为什么深度与关系不是二选一

Dense 或 sparse depth supervision 能够提供 metric scale，但单个深度值并不直接表达自由空间连通性、遮挡顺序和跨视角一致性。关系监督适合驾驶 VLM 的问题形式，但如果缺少视觉 metric grounding，模型可能只学习固定坐标或数据集先验。因此，GeoToken 将两者分工：

```text
Depth grounds metric perception.
Relations organize geometry for reasoning.
Language slots expose geometry to the VLM.
```

深度负责“看见多远”，关系负责“这些几何元素如何组成场景”，LGSS 负责“语言模型如何查询该场景”。

### 5.2 为什么不直接生成 RGB 深度图

将深度编码为可逆 RGB 图像非常适合原生图像生成器，因为它可以复用生成模型的输出空间。GeoToken 的目标是训练轻量 camera tokenizer，而不是训练大型图像生成器。因此，我们只借鉴连续、单调、近场敏感的 metric-depth transform，直接预测标量 warped depth。这样保留了关键思想，同时避免引入昂贵的生成模型和不必要的 RGB 编码。

### 5.3 为什么固定 anchor distance 不能证明三维感知

固定 BEV anchor 的中心距离和方向由坐标确定。若 query 中包含坐标，模型无需读取图像即可预测这些标签。因此，这类指标只能验证位置编码或几何条件是否可解码，不能单独证明 LiDAR 将三维能力蒸馏到相机。GeoToken 将主要证据建立在 surface depth、occupancy、topology、visibility、occlusion、zero/shuffle drop 和最终空间 QA 上。

### 5.4 为什么冻结 VLM

完整微调 VLM 可能掩盖收益来源。冻结 camera tokenizer 和 VLM、仅训练同容量 adapter，可以更直接地回答：预训练后的 camera token 是否比原始视觉 token 包含更可访问的空间信息。

---

## 6. 局限性

GeoToken 需要训练阶段存在同步 camera-LiDAR 数据，并依赖较准确的 ego pose 与标定。LiDAR 在远距离、恶劣天气和遮挡区域较稀疏，可能导致深度与关系标签不完整。多帧点云累积可以提高覆盖率，但动态对象会引入错位，需要更严格的 motion compensation 和 reliability mask。

Power-warp depth 的参数体现了近场优先假设，未必适用于所有道路速度和传感器范围，需要在验证集和跨数据集实验中分析。固定 BEV anchor 仍然携带空间先验，即使使用 visual residual，也不能完全排除模型利用数据集布局偏差，因此 coordinate-only、zero-image 和 shuffled-image 测试是必要但非充分的验证。

LGSS 使用自动生成的弱空间短语，仍可能存在模板偏差。Hard paraphrase、object composition 和 shuffled phrase 消融能够降低这一风险，但不能替代真实人类驾驶问答。最后，冻结 VLM 空间 QA 是较强代理任务，但不能替代闭环驾驶评测。

---

## 7. 结论

本文提出 GeoToken，一种面向自动驾驶空间推理的深度扎根 LiDAR 特权关系预训练框架。GeoToken 在训练阶段使用 LiDAR 提供连续 metric-depth、BEV occupancy、topology、visibility 和 occlusion 监督，在推理阶段仅使用多视角相机。

与仅做深度回归的方法不同，GeoToken 将局部 metric geometry 组织为适合语言推理的场景关系；与仅做固定 anchor 关系监督的方法不同，GeoToken 显式控制坐标捷径，并要求观测依赖能力在图像置零和跨样本打乱时显著下降。通过 LGSS 和冻结 VLM adapter，预训练空间 token 可以支持距离、方向、自由空间、遮挡、跨视角和时序问答。

本文的核心观点是：LiDAR 特权训练的价值不应体现为模型能够重新解码已知坐标，而应体现为纯相机 token 获得了随场景变化、依赖视觉证据且可被语言访问的三维结构。

---

## 参考文献

[1] nuScenes: A Multimodal Dataset for Autonomous Driving. CVPR, 2020.

[2] KITTI-360: A Novel Dataset and Benchmarks for Urban Scene Understanding in 2D and 3D. TPAMI, 2022.

[3] DriveLM: Driving with Graph Visual Question Answering. ECCV, 2024.

[4] SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities. CVPR, 2024.

[5] SimLingo: Vision-Only Closed-Loop Autonomous Driving with Language-Action Alignment. CVPR, 2025.

[6] VisionPAD: A Vision-Centric Pre-training Paradigm for Autonomous Driving. CVPR, 2025.

[7] SpaceDrive: Infusing Spatial Awareness into VLM-based Autonomous Driving. arXiv, 2025.

[8] RoboDriveVLM: A Benchmark and Baseline towards Robust Vision-Language Models for Autonomous Driving. arXiv, 2025.

[9] VLM-3R: Vision-Language Models Augmented with Instruction-Aligned 3D Reconstruction. CVPR, 2026.

[10] DRIVESPATIAL: A Benchmark for Spatiotemporal Intelligence in VLMs for Autonomous Driving. arXiv, 2026.

[11] Image Generators are Generalist Vision Learners. arXiv:2604.20329, 2026.

[12] Depth Anything V3: Recovering the Visual Space from Any Views. 2025.

[13] Metric3D v2: A Versatile Monocular Geometric Foundation Model for Zero-shot Metric Depth and Surface Normal Estimation. 2024.
