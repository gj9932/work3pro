# work3_1 Pro 方向更新：GeoDistill-VLM — 把 LiDAR 几何蒸馏到 Qwen2.5-VL 原生视觉 token

> 更新时间：2026-06-11
> 上版：[plan/work3pro.md](plan/work3pro.md)（GeoToken / PSRD / LGSS）
> 论文：[paper/work3pro_cvpr2027_draft5_zh_qwen.md](paper/work3pro_cvpr2027_draft5_zh_qwen.md)
>
> 本版根据 draft5 收敛：**冻结 Qwen2.5-VL 原生 Vision Encoder 与 merger，只训练 LiDAR-Privileged Geometry Adapter（LPGA）；推理移除 LiDAR 与外部深度网络，仅靠相机 + 标定 + ego state。**

---

## 0. 与 work3pro 的核心差异

| 维度 | work3pro（旧）| work3_1pro（新）|
|---|---|---|
| 视觉路径 | 自建 DINOv2 / SigLIP camera tokenizer | **冻结 Qwen2.5-VL 原生 Vision Encoder + spatial merger** |
| 主图节点 | 1600 candidate BEV anchors（`40 × 40` ego grid），训练用 `L=400` active | **Qwen `image_embeds` merged token，与原生 token 一一对应** |
| 几何核心 | PSRD：BEV anchor pairwise relation graph | **LPGA = R²AC depth + factorized allocation + sub-token ray + sparse relation residual** |
| 深度表示 | 隐含在 BEV occupancy / pairwise distance bin | **R²AC 连续可逆压扩，token 级 `a_p ∈ [0,1]` 由 risk × reliability 因子化** |
| 对接 VLM | LGSS slots + Q-Former projector | **3D PE Φ(ĉ) + relation residual ΔH_rel，零初始化门控直加到 `image_embeds`** |
| 语言访问 | LGSS（object/region/relation slots，Hungarian + 短语对比）| **不再保留 LGSS**；用 LoRA rank-16 微调 Qwen LLM；可选 `<POS>` coordinate decoder |
| 闭环驾驶 | 不作为第一篇必要项 | **Bench2Drive 闭环作为主表 5（必报）** |
| 推理依赖 | 仅相机（理想）| 仅相机 + 标定 + ego state；**不运行 LiDAR、不运行外部深度网络** |
| 主对手 | DINOv2 + depth/BEV auxiliary、SimLingo 概念 | **SpaceDrive-style + LiDAR-calibrated SpaceDrive** |

被弃用：BEV anchor 主图、LGSS slot 接口、Q-Former projector、object/region/relation 三类 slot 与 phrase contrastive。被吸收：稀疏 pair sampling、reliability mask、跨视角一致性、camera degradation 鲁棒性。

---

## 1. 一句话定位

**GeoDistill-VLM: distilling privileged LiDAR geometry into native Qwen2.5-VL visual tokens for camera-only driving.**

中文：**冻结 Qwen2.5-VL 原生视觉路径，只训练一个轻量适配器，把训练期 LiDAR 提供的 token 级 metric depth、跨视角与遮挡关系蒸馏为视觉 token 上的几何残差；部署时仅用相机、标定和 ego state。**

写作时不要说 token 等于 ego 坐标点。准确表述：Qwen merged visual tokens 在被 LLM 消费前，被叠加上 token 级预测三维位置编码与稀疏关系残差。

---

## 2. 研究问题

### 2.1 主问题

> 能否冻结 Qwen2.5-VL 原生 Vision Encoder 与 merger，仅训练一个轻量几何适配器，使 Qwen 原生视觉 token 在推理时携带可用于 VLM 空间推理与规划的度量、跨视角与遮挡几何，并完全移除 LiDAR 与外部深度网络？

### 2.2 子问题（决定章节边界）

1. token 级深度该用什么表示？— R²AC 连续压扩，灵敏度 `a_p` 由 risk × reliability 因子化（§3.5）。
2. token 区域内前景 vs 背景如何对齐？— foreground quantile + sub-token ray offset（§3.4 / 3.6）。
3. 几何信息如何写回 Qwen 原生 token，不破坏其语义？— 零初始化门控的 3D PE 与 relation residual（§3.7 / 3.10）。
4. 跨视角与遮挡关系如何在 `O(N²)` 边上避免爆炸？— sparse edge sampling + 同视角/跨视角分支独立 edge encoder（§3.8 / 3.9）。
5. 收益必须来自几何，而不是更多数据或 LoRA？— shuffle 系列因果验证（§5.7）。
6. 替代外部深度网络是否成立？— 必须由 30m+ 深度 + 空间 QA + 规划 + 延迟 Pareto 联合验证（§5.6 表 6）。

---

## 3. 模块拆分（与论文 §3 严格对齐）

### 3.1 冻结 Qwen 原生路径

```text
multi-view images
  → Qwen2_5_VisionTransformerPretrainedModel  (frozen)
  → spatial merger 2×2                        (frozen)
  → image_embeds  H_img ∈ R^{N × 3584}
```

每个 merged token 必须额外携带：

```text
meta_p = {camera_id, frame_id, image_grid_thw,
          token_center_uv (u_p, v_p), token_region R_p,
          K, T_ego→cam}
```

**不允许**：修改 `f_visual` / `g_merger`、增加从自定义视觉特征到 LLM 的 projector、改变 Qwen image processor。

### 3.2 LiDAR-Privileged Geometry Adapter（LPGA）

五个轻量组件，bottleneck `3584 → 512 → 512`：

| 组件 | 输入 | 输出 | 备注 |
|---|---|---|---|
| Token geometry encoder `A_token` | `LN(h_p), e_cam(p), e_uv(p)` | `g_p^mono` | 不接收 calibration，避免记忆数据集深度先验 |
| `A_geo` | `g_p^mono, e_calib(p)` | `g_p^geo` | 仅供 relation adapter 使用 |
| R²AC depth head `A_depth` | `g_p^mono` | `u_hat_p ∈ [0,1]` | sigmoid，压扩域 |
| Risk head `A_risk` | `g_p^mono, e_ego` | `r_hat_p ∈ [0,1]` | 唯一接收 ego state 的 head |
| Reliability head `A_reli` | `g_p^mono` | `q_hat_p ∈ [0,1]` | 学局部几何一致性，不学 LiDAR 采样密度 |
| Sub-token ray head `A_ray` | `g_p^mono` | `δ_hat_p ∈ [-1,1]^2` | tanh，区域归一化偏移 |
| Relation edge encoder `A_edge^same / A_edge^cross` | `g_p^geo, g_q^geo, ξ_pq` | `r_pq` | 同视角与跨视角分支参数不共享 |
| Relation projection `W_up` | `z_p^rel` | `Δh_p^rel ∈ R^{3584}` | 小方差非零初始化 `σ_up=1e-4` |

**强制约束**：
- ego state `e_ego` 只能进入 risk head；
- calibration `e_calib(p)` 只能进入 `A_geo` 与反投影；
- depth/allocation/ray head 不读 calibration。

### 3.3 Token-level LiDAR 标签构造（数据侧）

对每个 merged token 区域 `R_p`：

```text
Q_p = {(u_j, v_j, d_j, c_j^ego) | projected point j ∈ R_p}
D_p = {d_j | j ∈ Q_p}

d_p              = Quantile(D_p, q=0.1)            # foreground quantile
S_p              = {j ∈ Q_p | d_j ≤ d_p + τ_fg}    # foreground subset
(u_p^*, v_p^*)   = weighted_mean over S_p           # foreground centroid
δ_p^*            = normalize((u_p^*, v_p^*) − (u_p, v_p)) clipped to [-1,1]²

m_p              = 1[D_p ≠ ∅]
m_p^D            = m_p · 1[0 ≤ d_p ≤ D_max]
n_p              = |D_p|
s_p              = 1 − exp(−n_p / n_0)
q_p^geom         = exp(−MAD(D_p) / (τ_q · Median(D_p) + ε))
q_p^label        = s_p · q_p^geom
```

dynamic-box 内的点只用中心帧，static 点可做多帧 ego-motion 累积。`min depth` 与 mean depth 仅作消融。

### 3.4 R²AC 与因子化 allocation

```text
d_safe   = v_ego · t_react + v_ego² / (2·a_brake) + d_margin
r_p      = σ((d_safe − x_p^ego)/τ_r) · σ(x_p^ego/τ_front) · exp(−y_p^ego² / (2 w_corridor²))
a_p^*    = r_p · [η + (1 − η) · q_p^geom]                   # oracle allocation
a_hat_p  = r_hat_p · [η + (1 − η) · q_hat_p]                # 解析组合，不再单独回归 a

μ(a)     = exp(β·a) − 1
F(d; a)  = log(1 + μ(a)·d/D_max) / log(1 + μ(a))            # a→0 退化为 d/D_max
F_inv(u; a) = D_max · (exp(β·a·u) − 1) / (exp(β·a) − 1)
```

灵敏度比 `(dF/dd|_{d=0}) / (dF/dd|_{d=D_max}) = exp(β·a_p)`，由 `a_p` 解析控制。

**实现强制**：
- 小 `a` 使用极限 + `log1p` / `expm1`；
- `u_p^* = F(d_p; a_p^*)`，永远用 oracle `a_p^*` 构造目标，避免移动 target；
- `d_hat_p = F_inv(u_hat_p; StopGrad(a_hat_p))`，`StopGrad` 阻止 allocation 通过逆变换吸收 depth 误差。

### 3.5 损失函数（严格对齐论文 §3.5–3.9）

```text
w_p = m_p^D · (q_p^label + ε_q)
L_comp  = Σ w_p · SmoothL1(u_hat_p, u_p^*) / Σ w_p
L_r     = Σ m_p · SmoothL1(r_hat_p, r_p)   / Σ m_p
L_q     = Σ m_p · s_p · SmoothL1(q_hat_p, q_p^geom) / Σ m_p s_p
L_alloc = λ_r · L_r + λ_q · L_q,    λ_r + λ_q = 1
L_rank  = pairwise pairwise margin in metric domain (decoded d_hat)
L_depth = L_comp + λ_rank · L_rank
L_ray   = Σ w_p · SmoothL1(δ_hat_p, δ_p^*) / Σ w_p
L_coord = Σ w_p · Huber(c_hat_p^ego, c_p^lidar) / Σ w_p

L_rel   = λ_δ L_δ + λ_order L_order + λ_cross L_cross + λ_topo L_topo + λ_occ L_occ

L_keep  = 1 − cos( Pool(H_geo), StopGrad(Pool(H_img)) )
```

**禁止**：
- 不能在主路径上额外回归 `a_hat_p`（仅消融启用 `L_a^aux`）；
- 不能对 raw depth 同时使用 R²AC 与显式 risk reweighting `(1+κr_p)`（重复强化）；
- `L_topo` / `L_occ` 仅在 LiDAR + 地图 + ray 可视化通过质量检查后启用。

### 3.6 几何注入（关键耦合点）

```text
H_geo = H_img + α_pe · Φ(c_hat^ego) + α_rel · ΔH_rel

α_pe  = α_pe_max · tanh(β_pe),    β_pe  = 0  at init
α_rel = α_rel_max · tanh(β_rel),  β_rel = 0  at init
W_up  ~ N(0, σ_up²),              σ_up  = 1e-4
```

`W_up` 必须非零初始化、`α_rel` 必须严格零初始化，否则两侧梯度同时消失。3D PE 维度拆分为 `d_x = d_y = 1194, d_z = 1196`（和 = 3584，免 padding）。

---

## 4. 训练阶段（A0 → A1 → B → C）

| 阶段 | 训练 | 冻结 | 关键损失 |
|---|---|---|---|
| **A0**（前 `T_warm` = Stage A 前 5%）| `A_token`, `A_risk`, `A_reli`, `A_ray`, relation edge encoder & heads | Qwen 全部 | `λ_alloc·L_alloc + λ_ray·L_ray + λ_rel·L_rel`；`α_pe = α_rel = 0`，`W_up` 不接入 |
| **A1** | A0 全部 + `A_depth` | Qwen 全部 | `L_depth + L_A0 + λ_coord·L_coord` |
| **B** | A1 全部 + `W_up` + 两个 gate + Qwen LLM rank-16 LoRA + 可选 `<POS>` coordinate decoder | Qwen Vision Encoder + merger | `L_LM + λ_plan·L_plan + λ_geo·L_geo + λ_keep·L_keep` |
| **C**（可选） | B 模块 | 同 B | `L_robust = ‖Pool(H_geo^deg) − StopGrad(Pool(H_geo^full))‖₂` |

### 4.1 A0 的硬约束

- A0 不解码 metric depth，因此 `L_coord`、`L_rank`、`L_depth` 都不启用；
- relation edge encoder 在 A0 学到稳定 token relation feature 后才允许 `W_up` 在 B 阶段接入；
- 必须验证 `r_hat_p / q_hat_p / a_hat_p` 在 A0 末端与 oracle 的 MAE 与 Spearman 已稳定，否则不进入 A1。

### 4.2 模块状态对照（与论文 §3.13 一致）

```text
              A0          A1          B           Inference
Vision        frozen      frozen      frozen      use
Merger        frozen      frozen      frozen      use
LiDAR         label       label       low-w label REMOVED
UniDepth      —           —           —           NOT USED
LPGA depth    train       train       train       use
LPGA alloc    train       train       train       use
LPGA ray      train       train       train       use
Relation enc  train       train       train       use
W_up          NOT WIRED   NOT WIRED   train(σ_up) use
α_pe / α_rel  fixed 0     fixed 0     train       use
Qwen LLM      frozen      frozen      LoRA-16     use
Coord decoder —           —           optional    optional
```

---

## 5. 实验闭环（论文 §4 完整对齐，新增 Bench2Drive 闭环）

### 5.1 数据集

- **nuScenes**（主）：6-cam + LiDAR + ego + 标定 + 3D box + 地图。LiDAR/3D box 仅训练标签。
- **OmniDrive / DriveLM 风格 QA**：场景描述 + 空间问答 + 反事实 + 规划指令。
- **KITTI-360**：跨数据集 metric depth、跨视角几何、定位泛化。
- **Bench2Drive**（必报）：闭环驾驶。**主协议不使用 Bench2Drive-VL 额外语言标注**；如使用，必须独立成行。

### 5.2 实现配置

```yaml
base_vlm: Qwen2.5-VL-7B-Instruct
input_resolution: 640 × 640        # 与 SpaceDrive 主配置对齐
cameras: 6
merged_token_grid: 6 × ~22-23 × ~22-23   # 由 image_grid_thw 决定
geometry_bottleneck: 512
token_adapter_layers: 2
relation_neighbors_P: 16 or 32
D_max: 80m
exp_beta: 16                       # 最大灵敏度比
n_0: 3
tau_q: 0.1
eps_q: 0.05
lambda_r: 0.5
lambda_q: 0.5                      # 必须和为 1
lambda_alloc: 1.0
tau_fg: 1.0m
tau_d: 1.0m
eta: 0.5
t_react: 1.0s
a_brake: 4.0 m/s²
d_margin: 2.0m
tau_r: 2.0m
tau_front: 1.0m
w_corridor: 2.0m
cross_view_voxel: 0.5m
tau_depth: 1.0m
PE_freq_tau: 10000
gate_max_pe: 1.0
gate_max_rel: 1.0
T_warm: 5% of stage A
sigma_up: 1e-4
lora_rank: 16
gate_init_pe: 0
gate_init_rel: 0
```

需要做敏感性分析的：`t_react, a_brake, w_corridor, D_max, exp(beta)`。同时报告训练集 `d_safe` 分布。

### 5.3 主要评测任务

| 任务 | 指标 |
|---|---|
| Driving Spatial QA | overall / category（distance, direction, topology, occlusion, cross-view, temporal）/ metric tolerance / hard paraphrase / object-composition |
| Token 几何 probing | AbsRel, RMSE, δ1；分段 0-10 / 10-30 / 30+；`r̂/q̂/â` MAE & Spearman；risk-stratified AbsRel；relative 3D, depth ordering, cross-view matching, occlusion |
| nuScenes 开环规划 | L2 1s/2s/3s, avg L2, collision rate, intersection rate |
| Bench2Drive 闭环 | Driving Score, Success Rate, Route Completion, Infraction Score, collision/off-road/red-light frequency |
| 鲁棒性 | clean vs single-cam drop / multi-cam drop / front-only / occlusion / night / calibration noise — Robustness Ratio |
| 推理效率 | 参数量、显存、6-cam latency、FLOPs；vs `Qwen + UniDepth + 3D PE` |

### 5.4 Baselines（论文 §4.4，18 行）

1. Qwen2.5-VL-7B-Instruct 原始
2. Qwen2.5-VL + LoRA rank-16
3. SpaceDrive-style：冻结 UniDepthV2 + 3D PE，**无 LiDAR 校准**
4. **LiDAR-calibrated SpaceDrive**：UniDepthV2 + 与 LPGA 等量 LiDAR depth/coord losses 训练 calibration head（公平最强对手）
5. Qwen + raw-depth adapter
6. Qwen + log-depth adapter
7. Qwen + 固定 power-warp adapter
8. Qwen + 固定 log-companding adapter
9. Qwen + R²AC, `a = a_const`（训练与推理均固定）
10. Qwen + R²AC, risk-only target
11. Qwen + R²AC, reliability-only target
12. Qwen + R²AC, oracle target & oracle decode（压扩上限）
13. Qwen + relation adapter only
14. Qwen + R²AC depth + 3D PE，无 relation residual
15. Qwen + relation residual，无显式 3D PE
16. **完整 GeoDistill-VLM**
17. 完整方法但解冻 Qwen Vision Encoder
18. LLaVA-1.5-7B + CLIP ViT-L/14 变体

**控制变量**：所有 Qwen 变体使用相同视觉输入 / LoRA rank / 任务数据；所有可训练 depth adapter 获得相同 LiDAR depth+coord 监督；外部冻结深度网络同时报告无 LiDAR 与有 LiDAR 校准两版；3D box / occupancy / map 标签预算单独报告。

### 5.5 主表（draft5 §4.5）

- 表 1 Spatial QA × 8 列；
- 表 2 token 几何 probe × 8 列（含 `r̂/q̂/â` ρ 与 risk-stratified AbsRel）；
- 表 3 relation probe × 5 列；
- 表 4 nuScenes 开环规划 × 6 列；
- 表 5 Bench2Drive 闭环 × 5 列；
- 表 6 推理效率（参数量 / 相对 UniDepthV2-L 占比 / 显存 / latency / 30m+ AbsRel / QA）。

### 5.6 因子级消融（不可省）

- **Vision encoder 与注入位置**：冻结 vs 解冻最后 4 层 vs 全量解冻；merger 前 / merger 后 / 额外 projector / 直接作用于 `image_embeds`。
- **深度表示**：raw / log / 固定 power-warp / 固定 log-companding / R²AC `a=a_const` / risk-only / reliability-only / `L_a^aux` only / 主方法 + `L_a^aux`。
- **Allocation factor 设计**：joint sigmoid vs factorized 共享 encoder（主方法）vs factorized 独立 encoder；`L_q` SmoothL1 vs soft-target BCE；oracle target + predicted decode vs oracle decode；predicted `a_hat` 生成移动 target；无 warm-up；无 StopGrad；显式 risk reweighting。
- **`D_max` 与 `exp(β)`**；single-sweep vs multi-sweep；min vs 10% quantile。
- **Token 代表射线**：固定中心 / LiDAR 占心 / 预测 ray offset / 无 `L_ray`；`τ_fg` 扫描。
- **几何注入**：无 PE / 仅 PE / 仅 relation / PE + relation；固定 PE scale；learnable zero-init gate；无 `L_keep`。
- **关系监督**：依次去掉 `L_δ / L_order / L_cross / L_topo / L_occ`；同视角 vs 跨视角 vs hard-negative；dense vs sparse；mean / learned attention / teacher-confidence attention；**跨相机边错误使用 `(δu, δv)` 的对照实现**。
- **VLM 训练**：冻结 LLM / LoRA 8 / 16 / 32；仅 LM vs LM + geometry retention；digit-wise vs `<POS>` coordinate decoder。

### 5.7 防捷径与因果验证（论文 §4.7，必跑）

| 实验 | 操作 | 预期 |
|---|---|---|
| **Image shuffle** | batch 内打乱 `image_embeds`，保留 LiDAR/标定 | 几何性能必须显著下降 |
| **Calibration shuffle** | 图像不变，交换 K / `T_ego→cam` | 3D 坐标与跨视角性能显著下降 |
| **Depth-label shuffle** | 训练或评测打乱 `d_p` | 提升必须消失 |
| **Allocation-label shuffle** | 分别 shuffle `r_p / q_p^geom / a_p^*` | 近场深度、30m+ 深度、碰撞率、QA 分别降级 |
| **Constant-depth baseline** | 所有 token 使用固定深度或相机平均 | 显式 PE 收益必须高于此 baseline |
| **Relation-label shuffle** | 保留深度，shuffle cross-view / depth-order | relation residual 独立贡献必须消失 |

---

## 6. 仓库结构（基于现有 `work3pro-geotoken` 分支增量）

```text
configs/geodistill/
  geodistill_qwen25vl_nuscenes.yaml
  ablation_r2ac.yaml
  ablation_relation.yaml
  ablation_injection.yaml
  ablation_shortcut.yaml

dataset/geodistill/
  nuscenes_qwen_dataset.py        # 包 image_grid_thw / token_region / meta_p
  lidar_token_label.py            # foreground quantile / centroid / s_p / q_geom
  multi_sweep_lidar.py            # static ego-motion, dynamic center-only
  driving_qa_dataset.py
  bench2drive_dataset.py
  corruptions.py                  # cam drop / front-only / occlusion / cal noise

geodistill/
  models/
    qwen_visual_frozen.py         # 包装 image_embeds + meta
    lpga_token_encoder.py         # A_token / A_geo
    lpga_heads.py                 # depth / risk / reli / ray heads
    r2ac.py                       # F / F_inv / 极限分支
    relation_edge.py              # A_edge^same / A_edge^cross / W_up
    relation_aggregator.py        # confidence attention / mean / learned
    geometry_injection.py         # Φ(ĉ) + α_pe / α_rel gates / W_up
    coord_decoder.py              # 可选 <POS>
  geometry/
    risk_field.py                 # d_safe + r_p
    sub_token_ray.py
    pe_3d.py                      # d_x=d_y=1194, d_z=1196
    edge_sampler.py               # P_local / P_ray / P_cross / P_hard / P_far
    cross_view_label.py           # voxel + object-id + z-buffer
  losses/
    r2ac_loss.py                  # L_comp / L_rank / L_alloc / L_ray
    coord_loss.py                 # L_coord
    relation_loss.py              # L_δ / L_order / L_cross / L_topo / L_occ
    keep_loss.py                  # L_keep cosine pool
    robust_loss.py
  trainers/
    train_stage_a0.py
    train_stage_a1.py
    train_stage_b.py
    train_stage_c_robust.py
  eval/
    eval_token_geometry_probe.py
    eval_relation_probe.py
    eval_spatial_qa.py
    eval_open_loop_planning.py
    eval_bench2drive.py
    eval_robustness.py
    eval_efficiency.py
    shortcut_audit.py             # image / calib / depth / alloc / relation shuffle
  utils/
    distributed.py
    checkpoint.py
    logging.py
    visualization.py
```

废弃 / 不再继续：`modules/bev_anchor_queries.py`、`modules/spatial_slot_reader.py`、`modules/slot_hungarian_matcher.py`、`tools/lgss_loss.py`、`dataset/spatial_phrase_templates.py`。

---

## 7. 实施路线（六阶段）

### Phase 1 — Qwen 原生路径冻结读取
- `dataset/geodistill/nuscenes_qwen_dataset.py`：调 `processor`，缓存 `image_grid_thw / token_region / token_center_uv / K / T_ego→cam / S_ego`。
- `models/qwen_visual_frozen.py`：暴露 `image_embeds, meta_p`，支持 6-cam batch；冻结全部参数，验证 `requires_grad=False`。
- 验收：单 clip 前向得到 `H_img ∈ R^{N×3584}`，`N` 与 `image_grid_thw` 一致；token region 反算回像素后与原图对齐误差 < 1 px。

### Phase 2 — Token-level LiDAR 标签
- `lidar_token_label.py`：投影、可见性、foreground quantile、centroid、`s_p / q_geom / q_label`；`min depth` 与 mean 作 ablation。
- `multi_sweep_lidar.py`：static ego-motion 累积 + dynamic box mask。
- 验收：在 nuScenes mini split 上，`m_p` 覆盖率、平均 `n_p`、`q_p^geom` 直方图、`d_safe` 分布报告，并人工抽样 50 个 token 检查 foreground centroid 是否落在前景表面。

### Phase 3 — R²AC + Allocation + Ray（A0 子阶段）
- `r2ac.py`：`F / F_inv` + 小 `a` 极限路径 + 数值稳定性单元测试；
- `lpga_heads.py`：depth / risk / reliability / ray heads；
- `risk_field.py`：`d_safe / r_p`；
- `losses/r2ac_loss.py`：`L_comp / L_rank / L_alloc / L_ray`；
- `trainers/train_stage_a0.py`：仅 `λ_alloc·L_alloc + λ_ray·L_ray`（不含 depth），监控 `r_hat / q_hat / a_hat` 校准曲线；
- 验收：A0 末端 `r_hat / q_hat` MAE 与 Spearman 在验证集上稳定，进入 A1。

### Phase 4 — Coord 解码 + 关系监督（A1）
- `sub_token_ray.py`：`u_bar / v_bar` + 反投影到 ego；
- `coord_loss.py`：`L_coord`；
- `relation_edge.py / cross_view_label.py / edge_sampler.py / relation_loss.py`：实现稀疏关系监督，**同视角与跨视角分支独立 edge encoder**；
- `relation_aggregator.py`：confidence attention（主） + mean + learned attention 三种聚合做对照；
- `trainers/train_stage_a1.py`：在 A0 之上加入 `L_depth + λ_coord·L_coord + L_rel`；
- 验收：在 mini split 上 token AbsRel < 设定阈值；relation `cross_view` AUC 高于 chance。

### Phase 5 — 几何注入 + LoRA 微调（B）
- `geometry_injection.py`：`Φ(ĉ)` + 两个 zero-init gate + `W_up` 小方差初始化；强制初始 `α_pe = α_rel = 0`；
- `keep_loss.py`：`L_keep` 全局 pooling；
- `trainers/train_stage_b.py`：训练 LPGA + gate + `W_up` + Qwen LLM LoRA rank-16；可选 `<POS>` decoder；
- 验收：B 第 0 步与原始 Qwen 输出 token-wise 完全一致（验证零初始化退化）；训练后 spatial QA 全维度优于 LoRA-only。

### Phase 6 — 鲁棒（C）+ 评测闭环
- `train_stage_c_robust.py`：`L_robust` Pool 距离；
- `eval/*`：
  - `eval_spatial_qa.py`（含 hard paraphrase / object-composition split）；
  - `eval_token_geometry_probe.py`（必报 `r̂/q̂/â` 校准 + risk-stratified AbsRel）；
  - `eval_relation_probe.py`；
  - `eval_open_loop_planning.py`（nuScenes）；
  - `eval_bench2drive.py`（**主协议无 Bench2Drive-VL 数据**）；
  - `eval_robustness.py`；
  - `eval_efficiency.py`（含 vs `Qwen + UniDepth + 3D PE`）；
- `shortcut_audit.py`：6 项因果验证；
- 验收：`shortcut_audit` 全部通过（即 shuffle 后性能下降）；表 1–6 主结果 + 全消融 + 全因果 shuffle 数据齐备。

---

## 8. MVP 边界

第一篇 MVP 必须包含：

1. nuScenes 上完整训练 A0 → A1 → B；
2. token 几何 probing + relation probing；
3. spatial QA（含 hard paraphrase）；
4. nuScenes 开环规划；
5. Bench2Drive 闭环主协议；
6. clean vs camera-drop / front-only / night 鲁棒性；
7. 6 项 shortcut shuffle；
8. R²AC + factorized allocation + 关系监督的核心消融；
9. SpaceDrive-style + LiDAR-calibrated SpaceDrive 公平对比。

可暂缓但要写入 future work：

- KITTI-360 完整跨域；
- Bench2Drive-VL 增强训练协议；
- 解冻最后若干视觉层；
- 可学习风险场替代固定前向走廊；
- 多层 token depth 表达；

---

## 9. 写作必须避免的坑

1. 不要说 LPGA 在所有像素级深度基准上击败大型 UniDepth；只声明 token-level 几何 + 端到端任务 Pareto。
2. 不要说 token 等于 ego 坐标点；准确表述：原生 token 被叠加 token 级预测三维位置编码与稀疏关系残差。
3. 不要把 `α_pe` / `α_rel` 写成普通可学习标量；必须强调 `tanh(β)` 与零初始化退化为原始 Qwen。
4. 不要让 reliability `q_hat` 学 LiDAR 采样密度；只学局部几何一致性 `q_p^geom`。
5. 不要在主路径上叠加 `L_a^aux` 与 `L_r + L_q`；只在消融中比较。
6. `L_topo` / `L_occ` 不能在没有 ray 可视化质量检查的情况下默认启用。
7. 跨相机边不能用 `(δu, δv)`；必须用 `(ρ_p^ego, ρ_q^ego, ρ_q − ρ_p, camera_pair_id)`。
8. Bench2Drive 主协议不能与 Bench2Drive-VL 混合；混合结果独立成行。
9. `replace`/`drop` 旧 work3pro 中的 BEV anchor / LGSS / Q-Former 叙事，不要在 related work 中混入。
10. 所有几何收益必须有对应 shortcut shuffle 通过，否则不写入 contribution。

---

## 10. 贡献点（最终三条）

1. **R²AC**：将驾驶动力学风险与图像可预测的局部几何一致性因子化为 token 级连续压扩强度；解析单调可逆，灵敏度比由 `exp(β·a_p)` 闭式控制。
2. **LPGA on frozen Qwen2.5-VL native tokens**：在不修改 Vision Encoder / merger / projector 的前提下，将 metric depth + sub-token ray + 三维位置编码 + 稀疏跨视角关系残差通过零初始化门控写回原生 `image_embeds`；推理移除 LiDAR 与外部深度网络。
3. **Sparse cross-view / occlusion relation distillation on native VLM tokens**：标定感知边、跨相机独立 edge encoder、置信度归一化聚合，在 `O(NP)` 复杂度内蒸馏 LiDAR 跨视角对应、深度顺序与遮挡关系到 LLM 实际消费的视觉 token。

VLM projector / depth head / BEV head / coordinate decoder 一律不写为 contribution，作为接口实现。

---

## 11. 会话日志

### 2026-06-11 — direction switched to GeoDistill-VLM on frozen Qwen2.5-VL

- 放弃自建 camera tokenizer + BEV anchor + LGSS 路线，整体收敛到 draft5：冻结 Qwen2.5-VL 原生路径 + LPGA。
- 主图节点变更为 Qwen merged token；BEV anchor 主图与 1600/400 anchor 选择规则全部废弃。
- 主创新换为 R²AC 因子化 allocation 与 token-level 关系残差注入；LGSS 全部移除，由 LoRA + 可选 `<POS>` 取代。
- 实验主表新增 Bench2Drive 闭环；主对手切换为 SpaceDrive-style 与 LiDAR-calibrated SpaceDrive。
- 仓库结构以 `geodistill/` 为新主线，`work3pro-geotoken` 分支保留作为旧实现参考。

