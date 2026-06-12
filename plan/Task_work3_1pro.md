# work3_1 Pro 任务拆分：GeoDistill-VLM / LPGA / R²AC

> 分支：`work3pro-geotoken`（继续使用，可在内部建 `work3_1pro-geodistill` 子线）
>
> 日期：2026-06-11
>
> 目标：按 [plan/work3_1pro.md](plan/work3_1pro.md) 与 [paper/work3pro_cvpr2027_draft5_zh_qwen.md](paper/work3pro_cvpr2027_draft5_zh_qwen.md) 重构为：
>
> **冻结 Qwen2.5-VL 原生 Vision Encoder 与 spatial merger，只训练 LiDAR-Privileged Geometry Adapter（LPGA）；将 token 级预测三维位置编码与稀疏关系残差通过零初始化门控写回原生 `image_embeds`；推理移除 LiDAR 与外部深度网络。**

---

## 0. 总体边界

### 0.1 新工作核心

1. **Qwen2.5-VL frozen native path**
   - 输入：6-cam clip。
   - 输出：`H_img ∈ R^{N×3584}` 与 `meta_p`（`image_grid_thw`、`token_region`、`token_center_uv`、`K`、`T_ego→cam`、`S_ego`）。
   - 部署：Vision Encoder 与 merger 全程冻结。

2. **LPGA：LiDAR-Privileged Geometry Adapter**
   - R²AC 连续可逆压扩 depth；
   - factorized allocation：risk × reliability 解析组合；
   - sub-token ray offset；
   - sparse cross-view / occlusion relation residual。

3. **Geometry injection**
   - `H_geo = H_img + α_pe·Φ(ĉ) + α_rel·ΔH_rel`；
   - `α_pe / α_rel` 零初始化、`W_up` 小方差非零初始化。

4. **VLM tuning**
   - Qwen LLM rank-16 LoRA；
   - 可选 `<POS>` coordinate decoder（规划接口，不作贡献）。

5. **驾驶闭环**
   - nuScenes 开环 + **Bench2Drive 闭环（主表 5）**；主协议不混入 Bench2Drive-VL 增强数据。

### 0.2 不再作为主约束的内容

- 不再使用 BEV anchor 主图（`40 × 40 = 1600` candidate / `L=400` active 全部废弃）；
- 不再保留 LGSS（object / region / relation slots、Hungarian matching、phrase contrastive、slot diversity / temporal consistency 全部移除）；
- 不再训练自建 camera tokenizer（DINOv2 / SigLIP / CLIP-ViT 不作为视觉路径）；
- 不再实现 calibration-aware BEV anchor cross-attention；
- 不再以 PSRD 形式直接监督 token affinity matrix（关系监督改为稀疏 pairwise edge，以 token 级几何为节点，节点是 Qwen merged token，不是 BEV anchor）。

### 0.3 旧资产复用边界

| 旧文件 | 用途 |
|---|---|
| [dataset/NuScenesDataset.py](dataset/NuScenesDataset.py) | 多相机 + range 读取参考 |
| [tools/gen_info.py](tools/gen_info.py) | nuScenes info 生成 |
| [tools/gen_range.py](tools/gen_range.py) | LiDAR projection 参考 |
| [tools/runner.py](tools/runner.py) | trainer 脚手架参考 |
| [dataset/geotoken/nuscenes_clip_dataset.py](dataset/geotoken/nuscenes_clip_dataset.py) | clip schema 与 calibration 读取参考；**meta 必须扩展 `image_grid_thw / token_region`** |
| [dataset/geotoken/corruptions.py](dataset/geotoken/corruptions.py) | 鲁棒性 corruption 接口直接复用 |
| [geotoken/](geotoken/) 既有 `bev_grid` 等几何代码 | 仅作消融对照保留；主路径不依赖 |

新代码以 `geodistill/` 命名，不在 `geotoken/` 上硬改。

### 0.4 第三方代码：SpaceDrive

SpaceDrive 是论文 §4.4 baseline 3 / 4 的主对手，以 git submodule 形式集成（**MIT license**），仅作为 baseline 与基础设施来源。来源：[arxiv 2512.10719](https://arxiv.org/abs/2512.10719)、[github.com/zhenghao2519/SpaceDrive](https://github.com/zhenghao2519/SpaceDrive)。

**可直接 vendor / cherry-pick**：

| 用途 | 借鉴内容 | 我方仅做 |
|---|---|---|
| M1 数据 | `mmdet3d`-based nuScenes info pipeline、6-cam 顺序、`640×640` 预处理 | 扩展 `image_grid_thw / token_region / token_center_uv / S_ego` |
| M6 几何注入（仅 PE 部分）| Universal 3D Positional Encoder 维度 / `τ` 设计（与论文 §3.7 设计一致）| 加 `α_pe = α_pe_max·tanh(β_pe)` 零初始化门控；source 改为 LPGA 预测坐标；**自研 `geodistill/geometry/pe_3d.py`，独立实现以便消融** |
| M9 coord interface | `<POS>` token + MLP coordinate decoder + Huber waypoint loss | 仅作接口，**不写为 contribution** |
| M9 LoRA | LoRA 训练脚手架、Qwen2.5-VL-7B / LLaVA-1.5-7B 配置（直接对应 baseline 18） | rank-16 严格对齐论文 §4.2 |
| M12 评测 protocol | nuScenes 开环、Bench2Drive 闭环口径（Driving Score / Success Rate / Route Completion / Infraction Score）| **主协议禁止 Bench2Drive-VL 数据增强** |
| M13 baseline 3 | 直接跑 SpaceDrive-style 原版（无 LiDAR 校准）| — |
| M13 baseline 4 | 在 baseline 3 之上叠加 LPGA 等量 `L_depth + L_coord` 监督，训练轻量 calibration head | — |
| M12 efficiency | UniDepth 子模块（`unidepth`）测 UniDepthV2-L 参数量 / 显存 / 6-cam latency | 输出表 6 相对 UniDepthV2-L 占比 |

**禁止借鉴**（这些是本论文 contribution，借了即稀释贡献）：

- R²AC 连续可逆压扩；
- factorized risk × reliability allocation + StopGrad 解码；
- sub-token ray offset + foreground quantile / centroid；
- LPGA token encoder & heads（输入隔离规则）；
- sparse cross-view / occlusion relation residual + 同视角 / 跨相机独立 edge encoder；
- 零初始化 gate + `W_up` 小方差非零初始化；
- 几何注入与 `L_keep` 全局 pooling。

**集成方式**：

```text
third_party/SpaceDrive/                  # git submodule, pinned commit
  └── projects/configs/spacedrive/
  └── unidepth/                          # nested submodule
geodistill/baselines/
  ├── spacedrive_style.py                # baseline 3 wrapper
  └── spacedrive_calibrated.py           # baseline 4: + LPGA depth/coord losses
geodistill/geometry/pe_3d.py             # 自研 3D PE,对齐设计但独立实现
geodistill/models/coord_decoder.py       # 自研 <POS> decoder
THIRD_PARTY_LICENSES.md                  # 列出 MIT 来源 + commit hash
```

**禁止**：

- 不要把 SpaceDrive 整个仓库 fork 进来作为主项目骨架；只能作 submodule；
- 不要直接 import SpaceDrive 内部 PE 类型，必须自研以保证 §4.6 「固定 PE scale / learnable zero-init gate / 无 `L_keep`」消融可控；
- 不要把 `<POS>` decoder 写成 contribution；论文 §3.11 已声明「不作为主要创新」。

---

## 1. 目录规划

```text
third_party/
  SpaceDrive/                       # git submodule, MIT, pinned commit
    └── unidepth/                   # nested submodule

configs/geodistill/
  geodistill_qwen25vl_nuscenes.yaml
  ablation_r2ac.yaml
  ablation_relation.yaml
  ablation_injection.yaml
  ablation_shortcut.yaml
  bench2drive.yaml
  baseline_spacedrive_style.yaml
  baseline_spacedrive_calibrated.yaml

dataset/geodistill/
  nuscenes_qwen_dataset.py         # 复用 nuscenes_clip_dataset，包 image_grid_thw
  lidar_token_label.py             # foreground quantile / centroid / s_p / q_geom
  multi_sweep_lidar.py             # static ego-motion 累积 + dynamic center-only
  driving_qa_dataset.py            # DriveLM/OmniDrive style + paraphrase / composition split
  bench2drive_dataset.py
  corruptions.py                   # 复用 dataset/geotoken/corruptions.py

geodistill/
  baselines/
    spacedrive_style.py            # baseline 3：SpaceDrive 原版 wrapper（无 LiDAR 校准）
    spacedrive_calibrated.py       # baseline 4：SpaceDrive + LPGA 等量 LiDAR depth/coord 监督
  models/
    qwen_visual_frozen.py          # 冻结 Vision Encoder + merger,暴露 image_embeds + meta
    lpga_token_encoder.py          # A_token / A_geo
    lpga_heads.py                  # depth / risk / reliability / ray heads
    r2ac.py                        # F / F_inv / 极限分支
    relation_edge.py               # A_edge^same / A_edge^cross
    relation_aggregator.py         # confidence attention / mean / learned
    geometry_injection.py          # Φ(ĉ) + α_pe / α_rel gates + W_up
    coord_decoder.py               # <POS>，对齐 SpaceDrive 接口但自研实现
  geometry/
    risk_field.py                  # d_safe + r_p
    sub_token_ray.py               # u_bar / v_bar + 反投影到 ego
    pe_3d.py                       # d_x = d_y = 1194, d_z = 1196，自研，不 import SpaceDrive
    edge_sampler.py                # P_local / P_ray / P_cross / P_hard / P_far
    cross_view_label.py            # voxel + object-id + z-buffer
  losses/
    r2ac_loss.py                   # L_comp / L_rank / L_alloc / L_ray
    coord_loss.py                  # L_coord
    relation_loss.py               # L_δ / L_order / L_cross / L_topo / L_occ
    keep_loss.py                   # L_keep cosine pool
    robust_loss.py                 # L_robust
  trainers/
    train_stage_a0.py
    train_stage_a1.py
    train_stage_b.py
    train_stage_c_robust.py
    train_baseline_spacedrive.py   # 跑 baseline 3 / 4
  eval/
    eval_token_geometry_probe.py
    eval_relation_probe.py
    eval_spatial_qa.py
    eval_open_loop_planning.py     # 借鉴 SpaceDrive 评测口径
    eval_bench2drive.py            # 借鉴 SpaceDrive 评测口径，主协议禁 VL 数据
    eval_robustness.py
    eval_efficiency.py             # 含 UniDepthV2-L 参数 / 显存 / latency
    shortcut_audit.py
  utils/
    distributed.py
    checkpoint.py
    logging.py
    visualization.py

scripts/geodistill/
  init_third_party.sh              # 拉 SpaceDrive submodule（含 unidepth）
  build_token_label_cache.py
  build_relation_label_cache.py
  build_driving_qa.py
  train_stage_a0.sh
  train_stage_a1.sh
  train_stage_b.sh
  train_stage_c.sh
  train_baseline_spacedrive.sh
  eval_all.sh
  shortcut_audit.sh
  collect_tables.py

THIRD_PARTY_LICENSES.md            # 标注 SpaceDrive MIT + commit hash
```

---

## 2. Milestone 拆分

### M0. 工程骨架与 Qwen 接入

目标：建立 `geodistill/` 包，确认 Qwen2.5-VL-7B-Instruct 在本地可冻结跑通 `image_embeds`，并初始化 SpaceDrive submodule。

任务：
- [ ] 添加 `transformers` 依赖（与 Qwen2.5-VL 兼容版本，写入 `requirements.txt`）。
- [ ] **初始化第三方代码**（一次性）：
  - `git submodule add https://github.com/zhenghao2519/SpaceDrive third_party/SpaceDrive`；
  - `git submodule update --init --recursive`（拉 nested `unidepth`）；
  - 记录 commit hash 到 `THIRD_PARTY_LICENSES.md`，注明 MIT；
  - `scripts/geodistill/init_third_party.sh` 自动化以上步骤。
- [ ] 创建最小骨架（仅当前阶段必要）：
  - `configs/geodistill/geodistill_qwen25vl_nuscenes.yaml`
  - `geodistill/__init__.py`
  - `geodistill/models/qwen_visual_frozen.py`
  - `dataset/geodistill/nuscenes_qwen_dataset.py`
  - `scripts/geodistill/build_token_label_cache.py`
- [ ] yaml 字段：dataset root / camera order / clip length T / image resolution `640×640` / Qwen ckpt path / `D_max` / `exp(β)` / `n_0` / `τ_q` / `ε_q` / `λ_r:λ_q` / `λ_alloc` / `τ_fg` / `τ_d` / `η` / RSS 参数 / `τ_r / τ_front / w_corridor` / 跨视角 voxel / `τ_depth` / PE `τ` / gate max / `T_warm` / `σ_up` / LoRA rank / gate init。

输出：可 import `geodistill`；可读取 yaml 并打印完整字段；`qwen_visual_frozen.py` 可在 mini batch 跑通 `image_embeds`，`requires_grad=False`；`third_party/SpaceDrive` 可成功 clone 且 `unidepth` 子模块就位。

验收：
- 单 sample 前向得到 `H_img ∈ R^{N × 3584}`，`N` 与 `image_grid_thw` 一致；
- 验证 Qwen Vision Encoder 与 merger 所有参数 `requires_grad=False`；
- 显存与 latency 对齐 SpaceDrive 配置（`640×640`、6-cam）；
- `THIRD_PARTY_LICENSES.md` 记录 SpaceDrive + UniDepth 的 commit hash 与 license。

---

### M1. nuScenes Qwen-aware Dataset

目标：基于 [dataset/geotoken/nuscenes_clip_dataset.py](dataset/geotoken/nuscenes_clip_dataset.py) 扩展 Qwen image processor 输出与 merged token meta。

任务：
- [ ] `dataset/geodistill/nuscenes_qwen_dataset.py`：
  - 调用 Qwen 原生 `processor`（不修改、不替换）；
  - 缓存 `pixel_values / image_grid_thw / token_center_uv / token_region / K / T_ego→cam / S_ego`；
  - 训练阶段额外暴露 `lidar_path / boxes_3d / map_labels / ego_pose_t→t0`；
  - 推理 schema 严格不含 LiDAR / 3D box / 地图。
- [ ] 6-cam 顺序按 SpaceDrive 配置统一：`CAM_FRONT / CAM_FRONT_RIGHT / CAM_BACK_RIGHT / CAM_BACK / CAM_BACK_LEFT / CAM_FRONT_LEFT`。
- [ ] 复用 [dataset/geotoken/corruptions.py](dataset/geotoken/corruptions.py)：cam drop / multi-cam drop / front-only / occlusion / calibration noise / low-light。

输出：6-cam dataloader；clean & corrupt 双视图 dataloader；token_region 反算回像素的工具。

验收：
- batch shape：`pixel_values[T, 6, 3, H, W]`、`image_grid_thw[T, 6, 3]`；
- 单 token 区域反算回原图，误差 ≤ 1 pixel；
- corruption 不改变 batch schema。

---

### M2. Token-Level LiDAR Label Builder

目标：把 LiDAR 点投影到每个 merged token 区域，构造 R²AC 训练标签。**M2 没通过 sanity check 不进入 M4 / M5。**

任务：
- [ ] `dataset/geodistill/lidar_token_label.py`：
  - LiDAR → ego → camera 投影（含时间同步、可见性过滤、动态错位剔除）；
  - 在 `R_p` 内聚合点：`Q_p / D_p`；
  - `d_p = Quantile(D_p, q=0.1)`；同时缓存 `min(D_p)` 与 `mean(D_p)` 供消融；
  - 前景子集 `S_p = {j ∈ Q_p | d_j ≤ d_p + τ_fg}`；
  - foreground centroid `(u_p^*, v_p^*)`，weighted by 时间距离 + 投影置信度；
  - sub-token offset target `δ_p^* = clip([2·(u_p^* − u_p)/W_R, 2·(v_p^* − v_p)/H_R], -1, 1)`；
  - `m_p = 1[D_p ≠ ∅]`、`m_p^D = m_p · 1[0 ≤ d_p ≤ D_max]`；
  - `n_p = |D_p|`、`s_p = 1 − exp(−n_p/n_0)`、`q_p^geom = exp(−MAD/(τ_q·Median+ε))`、`q_p^label = s_p · q_p^geom`；
  - teacher 坐标 `c_p^lidar = T_cam→ego · d_p · K^{-1} [u_p^*, v_p^*, 1]^T`。
- [ ] `multi_sweep_lidar.py`：static 点 ego-motion 累积；dynamic-box 内的点只用中心帧。
- [ ] `scripts/geodistill/build_token_label_cache.py`：mini split 上预生成 token-level cache。

输出：每 sample 缓存 `{d_p, m_p, m_p^D, s_p, q_p^geom, q_p^label, δ_p^*, c_p^lidar}`，与 Qwen merged token 一一对应。

验收（5 类 sanity check 可视化）：
1. LiDAR points overlay 在原图与 merged token region；
2. `d_p` 直方图（含 `m_p^D` 比例）；
3. `q_p^geom`、`s_p`、`q_p^label` 分布；
4. foreground centroid `(u_p^*, v_p^*)` 在前景表面（人工抽样 50 个 token）；
5. `c_p^lidar` 三维点云 vs 原始 LiDAR 对齐。

---

### M3. R²AC + Risk Field

目标：实现连续可逆压扩与 RSS 风险场，含数值稳定路径。

任务：
- [ ] `geodistill/models/r2ac.py`：
  - `F(d, a) = log1p(μ·d/D_max) / log1p(μ)`，`μ = expm1(β·a)`；
  - `F_inv(u, a) = D_max · expm1(β·a·u) / expm1(β·a)`；
  - 当 `|a| < a_eps` 走线性极限 `F = d/D_max`、`F_inv = D_max·u`；
  - 单元测试：`F(F_inv(u,a),a) ≈ u`、`a→0` 极限连续、严格单调；
  - 验证灵敏度比 `(dF/dd|0)/(dF/dd|D_max) = exp(β·a)`。
- [ ] `geodistill/geometry/risk_field.py`：
  - `d_safe = v_ego·t_react + v_ego²/(2·a_brake) + d_margin`；
  - `r_p = σ((d_safe − x_p^ego)/τ_r) · σ(x_p^ego/τ_front) · exp(−y_p^ego²/(2·w_corridor²))`；
  - 输出 `r_p, a_p^* = r_p · [η + (1 − η)·q_p^geom]`；
  - 默认参数：`t_react=1.0s, a_brake=4.0 m/s², d_margin=2.0m, τ_r=2.0m, τ_front=1.0m, w_corridor=2.0m, η=0.5`。
- [ ] `D_max = 80m`、`β = log(16)`，灵敏度比上限 16；提供敏感性扫描接口。

验收：
- 单元测试覆盖极限、可逆、单调、灵敏度比；
- mini split 上 `d_safe` 与 `r_p / a_p^*` 分布报告输入 §5.2 sensitivity 实验。

---

### M4. LPGA Token Encoder & Heads

目标：实现 §3.2 五个 LPGA 组件，**严格遵守输入隔离**：
- `A_token` 只读 `LN(h_p), e_cam, e_uv`；
- `A_geo = MLP([g_mono, e_calib])`；
- depth / allocation / ray head 只读 `g_p^mono`；
- risk head 是唯一额外读 `e_ego` 的 head；
- calibration 不进 depth / allocation / ray，不进 risk。

任务：
- [ ] `geodistill/models/lpga_token_encoder.py`：bottleneck `3584 → 512 → 512`；2 层 MLP 或 2 层 Transformer。
- [ ] `geodistill/models/lpga_heads.py`：
  - `A_depth`：`g_mono → sigmoid → u_hat_p`；
  - `A_risk`：`[g_mono, e_ego] → sigmoid → r_hat_p`；
  - `A_reli`：`g_mono → sigmoid → q_hat_p`；
  - `A_ray`：`g_mono → tanh → δ_hat_p`；
  - 内部计算 `a_hat_p = r_hat_p · [η + (1 − η)·q_hat_p]`，**不再加冗余 a 回归头**。
- [ ] `geodistill/geometry/sub_token_ray.py`：`u_bar / v_bar` + 反投影到 ego。

验收：
- 自动化 assert 输入张量类型，确保 calibration 不流入 depth / allocation / ray，risk 唯一接收 `e_ego`；
- forward shape 检查；
- 在 mini batch 上前向稳定不溢出。

---

### M5. R²AC Loss + Coord Loss

任务：
- [ ] `geodistill/losses/r2ac_loss.py`：
  - `w_p = m_p^D · (q_p^label + ε_q)`；
  - `L_comp = Σ w_p · SmoothL1(u_hat_p, u_p^*) / Σ w_p`，`u_p^* = F(d_p, a_p^*)`（始终用 oracle `a_p^*`）；
  - `L_r = Σ m_p · SmoothL1(r_hat_p, r_p) / Σ m_p`；
  - `L_q = Σ m_p · s_p · SmoothL1(q_hat_p, q_p^geom) / Σ m_p s_p`；
  - `L_alloc = λ_r·L_r + λ_q·L_q`，强制 `λ_r + λ_q = 1`；
  - `L_rank` 在 metric domain：`d_hat_p = F_inv(u_hat_p, StopGrad(a_hat_p))`；pair `(p, q)` 仅取同视角邻域或可靠跨视角对应；
  - `L_ray = Σ w_p · SmoothL1(δ_hat_p, δ_p^*) / Σ w_p`。
- [ ] `geodistill/losses/coord_loss.py`：
  - `L_coord = Σ w_p · Huber(c_hat_p^ego, c_p^lidar) / Σ w_p`；
  - teacher 坐标用 foreground centroid 射线。
- [ ] 严格禁止：
  - 主路径上回归 `a_hat_p`；
  - 主路径上对 raw / log depth 同时叠加 `(1 + κ·r_p)` reweighting；
  - allocation 通过 `F_inv` 接收 depth 误差梯度（必须 StopGrad）。
- [ ] 提供 `L_a^aux` 仅作消融。

验收：
- 单元测试覆盖 `λ_r + λ_q = 1` 强制；
- StopGrad 验证（在 `d_hat` 上做反向，`a_hat_p` 的梯度为 0）；
- 单 batch overfit 可下降。

---

### M6. Sparse Relation Adapter

目标：实现 §3.8–3.10 的稀疏关系蒸馏与 geometry residual 注入。

任务：
- [ ] `geodistill/geometry/cross_view_label.py`：
  - 体素化前景子集 `S_p` 到 ego；
  - 跨相机候选边只在相邻相机重叠 FOV + 三维近邻产生；
  - `cross_view = 1` 当 (a) 同 voxel + 深度差 < `τ_depth`，或 (b) 同 3D object id + 双向可见性通过；
  - 稀疏 z-buffer 可见性检查；
  - 边可靠度 `e_conf = q_p^label · q_q^label · voxel_overlap · depth_consistency`。
- [ ] `geodistill/geometry/edge_sampler.py`：
  - 每 token 最多 `P` 条边（默认 16/32）；
  - `P_local` 同视角邻；`P_ray` 同 / 邻 ray bin；`P_cross` 跨相机正样本；`P_hard` 外观相似但 3D 不同的负样本；`P_far` 随机远距离 valid token。
- [ ] `geodistill/models/relation_edge.py`：**同视角与跨相机分支独立**
  - 同视角 `xi_pq^same = [δu/W, δv/H, ρ_q^ego − ρ_p^ego]`；
  - 跨相机 `xi_pq^cross = [ρ_p^ego, ρ_q^ego, ρ_q − ρ_p, camera_pair_id_emb]`；
  - 跨相机分支**禁止**接收 `(δu, δv)`；
  - 输出 `r_pq` + 多任务 head：`Δc_pq, depth_order, cross_view, topology, occlusion`。
- [ ] `geodistill/models/relation_aggregator.py`：
  - 主：`ω_pq = softmax(A_conf(r_pq) + log(e_conf+ε))`，`z_p^rel = Σ ω_pq · V_rel·r_pq`；
  - 消融：mean、无 teacher confidence 的 attention。
- [ ] `geodistill/losses/relation_loss.py`：
  - `L_δ = Huber, L_order = CE, L_cross = BCE, L_topo = CE, L_occ = CE`；
  - 仅 reliability mask 有效边参与 loss；
  - 第一阶段默认启用 `L_δ + L_order + L_cross`，`L_topo / L_occ` 在 ray 可视化质量检查通过后启用。
- [ ] `geodistill/models/geometry_injection.py`：
  - `Φ(ĉ)`，`d_x=d_y=1194, d_z=1196`，PE `τ=10000`；**自研，不 import SpaceDrive**（保证消融可控）；
  - 与 SpaceDrive PE 设计对齐的部分（维度拆分 / sinusoidal / `τ`）记录在文件 docstring，便于审稿对比；
  - `α_pe = α_pe_max·tanh(β_pe)`、`β_pe = 0` at init；
  - `α_rel = α_rel_max·tanh(β_rel)`、`β_rel = 0` at init；
  - `W_up ~ N(0, σ_up²), σ_up=1e-4`（**不可零初始化**）；
  - 输出 `H_geo = H_img + α_pe·Φ(ĉ) + α_rel·ΔH_rel`。

验收：
- 跨相机边特征与同视角边特征字段完全不同（构造期 assert）；
- 初始 forward `H_geo == H_img`（两 gate 严格为 0）；
- B 阶段第 1 步起，`α_pe / α_rel` 接收非零梯度（验证 `W_up` 非零初始化）。

---

### M7. 阶段 A0：allocation + ray + relation 热启动

任务：
- [ ] `geodistill/trainers/train_stage_a0.py`：
  - 冻结 Qwen 全部；
  - 训练 `A_token, A_risk, A_reli, A_ray, relation edge encoder, relation heads`；
  - 损失：`L_A0 = λ_alloc·L_alloc + λ_ray·L_ray + λ_rel·L_rel`；
  - `α_pe = α_rel = 0`，`W_up` 不接入；
  - `T_warm = stage A 前 5% steps`；
  - 监控：`r_hat / q_hat / a_hat` MAE & Spearman、relation cross-view AUC、edge sampler 类别覆盖。
- [ ] AMP / grad clip / resume / checkpoint / wandb（可选）。

验收：
- 单 batch overfit 损失下降；
- 验证集上 `r_hat / q_hat` 相对 oracle 的 Spearman ≥ 阈值才进入 A1；
- relation cross-view AUC 显著高于 chance。

---

### M8. 阶段 A1：metric 解码 + coord loss

任务：
- [ ] `geodistill/trainers/train_stage_a1.py`：
  - 在 A0 基础上加入 `A_depth`；
  - 损失：`L_A1 = L_depth + L_A0 + λ_coord·L_coord`；
  - `L_depth = L_comp + λ_rank·L_rank`；
  - `d_hat_p = F_inv(u_hat_p, StopGrad(a_hat_p))`，必须验证 StopGrad 行为；
  - relation supervision 仍只训练 standalone edge encoder + heads，`W_up` 不接入。
- [ ] 报告 token 级 AbsRel / RMSE / δ1，分段 0–10 / 10–30 / 30+，risk-stratified AbsRel（按 `a_p^*` 分位）。

验收：
- mini split token AbsRel 低于 raw-depth adapter；
- 30+ AbsRel 不出现不可接受的退化；
- relation cross-view / depth-order 指标稳定。

---

### M9. 阶段 B：几何注入 + Qwen LLM LoRA

任务：
- [ ] `geodistill/trainers/train_stage_b.py`：
  - 训练：A1 全部 + `W_up`（小方差初始化）+ 两个 gate（零初始化）+ Qwen LLM rank-16 LoRA + 可选 `<POS>` coordinate decoder；
  - 冻结：Vision Encoder + merger；
  - **LoRA 脚手架可借鉴 SpaceDrive `projects/configs/spacedrive/spacedrive_qwen.py` 的 LoRA 设置**（rank、target_modules、lr 比例），但 rank 严格固定 16；
  - 损失：`L_B = L_LM + λ_plan·L_plan + λ_geo·L_geo + λ_keep·L_keep`；
  - `L_keep = 1 − cos(Pool(H_geo), StopGrad(Pool(H_img)))`，`Pool` 先按 cam mean、再按 cam 平均；
  - 几何损失保留低权重避免漂移。
- [ ] `geodistill/models/coord_decoder.py`：`<POS>` 后的 hidden state 接 MLP，Huber loss 回归 waypoints；**接口与 SpaceDrive `<POS>` 对齐**（特殊 token id、hidden state 取位、Huber 形式），实现自研，写入文件 docstring。
- [ ] 数据：DriveLM-nuScenes / OmniDrive 风格 QA + 自构造 spatial QA。

验收：
- B 第 0 步 `H_geo == H_img`（gate 零初始化 + LoRA 零初始化默认行为）；
- Qwen2.5-VL + LoRA-only baseline 与本阶段同条件训练，作为对照；
- spatial QA 全维度优于 LoRA-only。

---

### M10. 阶段 C：Robustness Distillation（可选但推荐）

任务：
- [ ] `geodistill/trainers/train_stage_c_robust.py`：
  - 完整输入做 teacher，退化输入做 student：`L_robust = ‖Pool(H_geo^deg) − StopGrad(Pool(H_geo^full))‖₂`；
  - corruption：cam drop / multi-cam drop / front-only / image occlusion / calibration noise / low-light。

验收：clean vs corrupt 下 spatial QA 与 token AbsRel 的 Robustness Ratio 显著高于无 C 阶段。

---

### M11. Driving Spatial QA 数据构造

目标：构造与 DriveLM / OmniDrive 风格对齐的空间 QA，含 hard paraphrase / object-composition / template split。

任务：
- [ ] `dataset/geodistill/driving_qa_dataset.py`：
  - 类别：metric distance / relative direction / object comparison / free-space topology / occlusion / cross-view correspondence / temporal motion / counterfactual trajectory；
  - 每条 QA 字段：`question, answer, type, involved_object_ids, source_relation, reliability, split_id`；
  - hard paraphrase split：训练短语 vs 测试问句不同表达；
  - object-composition split：训练对象组合 vs 测试组合分离；
  - template split：同 relation 不同 scene；
  - reliability 过滤：sparse LiDAR / ambiguous occlusion / unstable dynamic association / 远距离小目标。
- [ ] `scripts/geodistill/build_driving_qa.py`。

输出：`spatial_qa_{train,val,test_template,test_paraphrase,test_composition}.jsonl`。

验收：每类 QA 数量统计；hard paraphrase 不与训练模板共享；answer 可由 relation graph / 3D box 重新核验。

---

### M12. 评测套件（论文 §4.5 表 1–6）

任务（每个 eval 输出统一 JSON：`method / config / checkpoint / split / metrics`）：

- [ ] `eval_token_geometry_probe.py`：冻结 Qwen + LPGA，训练同容量 linear / 2-layer probe；指标 AbsRel / RMSE / δ1，分段 0–10 / 10–30 / 30+，`r̂/q̂/â` MAE & Spearman、risk-stratified AbsRel、relative 3D、depth ordering、cross-view matching、occlusion accuracy。
- [ ] `eval_relation_probe.py`：relative 3D ↓ / order ↑ / cross-view ↑ / occlusion ↑ / topology ↑。
- [ ] `eval_spatial_qa.py`：overall / 各 category / metric tolerance / hard paraphrase / object-composition。
- [ ] `eval_open_loop_planning.py`（nuScenes）：L2 1s/2s/3s, avg L2, collision, intersection；**评测口径直接对齐 SpaceDrive `tools/` 的 nuScenes planning eval**，确保数字可比。
- [ ] `eval_bench2drive.py`：Driving Score, Success Rate, Route Completion, Infraction Score, collision/off-road/red-light frequency；**评测口径直接对齐 SpaceDrive Bench2Drive eval**；**主协议禁止使用 Bench2Drive-VL 数据**，如使用必须以独立行报告。
- [ ] `eval_robustness.py`：clean / cam drop / multi-cam drop / front-only / occlusion / night / calibration noise；输出 Robustness Ratio。
- [ ] `eval_efficiency.py`：参数量、显存、6-cam latency、FLOPs；与 `Qwen + UniDepthV2-L + 3D PE` 和 `LiDAR-calibrated SpaceDrive` 对比；同时报告 30+ AbsRel 与 spatial QA。

验收：所有 eval 脚本输出统一 JSON；表 6 必须报告 Pareto（不能只报 latency）。

---

### M13. Baselines（论文 §4.4，18 行）

任务：
- [ ] 实现 / 复用 18 个 baseline，全部使用同一视觉输入、LoRA rank、任务数据：
  1. Qwen2.5-VL-7B 原始
  2. Qwen + LoRA rank-16
  3. **SpaceDrive-style**：冻结 UniDepthV2 + 3D PE，无 LiDAR 校准；
     - 通过 `geodistill/baselines/spacedrive_style.py` 调 `third_party/SpaceDrive` 的 `spacedrive_qwen.py` config（pinned commit）；
     - 不修改 SpaceDrive 内部代码，只 wrap I/O 与评测，确保 baseline 与公开实现一致；
     - 报告时注明使用的 SpaceDrive commit hash 与 HF 权重 version。
  4. **LiDAR-calibrated SpaceDrive**：UniDepthV2 + 与 LPGA 等量 LiDAR depth/coord losses 训练 calibration head；
     - 通过 `geodistill/baselines/spacedrive_calibrated.py`：在 SpaceDrive UniDepthV2 输出之后接 `geodistill/losses/r2ac_loss.py` 的 `L_comp`（不启用 R²AC factorized allocation，避免污染 R²AC contribution）+ `L_coord`；
     - calibration head 容量与 LPGA bottleneck 等量（512）。
  5. raw-depth adapter
  6. log-depth adapter
  7. 固定 power-warp adapter
  8. 固定 log-companding adapter
  9. R²AC, `a = a_const`
  10. R²AC, risk-only
  11. R²AC, reliability-only
  12. R²AC, oracle target & oracle decode
  13. relation adapter only
  14. R²AC depth + 3D PE，无 relation residual
  15. relation residual，无显式 3D PE
  16. **完整 GeoDistill-VLM**
  17. 完整方法 + 解冻 Qwen Vision Encoder
  18. **LLaVA-1.5-7B + CLIP ViT-L/14 变体**：直接复用 SpaceDrive `spacedrive_llava.py` 的 LLaVA wrapper（pinned commit），用作 Vision Encoder 选择敏感性对照。
- [ ] 所有可训练 depth adapter 获得相同 LiDAR depth + coord 监督；
- [ ] 外部冻结深度网络同时报告无 LiDAR 与有 LiDAR calibration 两版；
- [ ] 3D box / occupancy / 地图标签预算单独报告。

验收：每个 baseline 都能独立 train + eval，输出可填表 1–6；baseline 3 / 4 / 18 在 README 注明 SpaceDrive 出处与 commit hash。

---

### M14. 因子级消融（论文 §4.6）

任务：
- [ ] **Vision encoder 注入位置**：冻结 visual + merger / 解冻最后 4 层 / 全量解冻 / merger 前注入 / merger 后注入 / 额外 projector / 直接作用于 `image_embeds`。
- [ ] **深度表示**：raw / log / 固定 power-warp / 固定 log-companding / R²AC `a_const` / risk-only / reliability-only / `L_a^aux` only / 主方法 + `L_a^aux` / joint allocation / factorized 共享 encoder（主方法）/ factorized 独立 encoder / `L_q` SmoothL1 vs soft-target BCE / oracle target + predicted decode / oracle target + oracle decode / predicted `a` 移动 target / 无 warm-up / 无 StopGrad / 显式 risk reweighting / 不同 `D_max` / `exp(β)` / single-sweep vs multi-sweep / min vs 10% quantile depth。
- [ ] **Token 代表射线**：固定中心 / LiDAR 占心（仅 teacher）/ 预测 ray offset / 无 `L_ray` / 不同 `τ_fg`。
- [ ] **几何注入**：无 PE / 仅 PE / 仅 relation / PE + relation / 固定 PE scale / learnable zero-init gate / 无 `L_keep`。
- [ ] **关系监督**：依次去掉 `L_δ / L_order / L_cross / L_topo / L_occ` / 同视角 / 跨视角 / hard negative / dense vs sparse / mean / learned attention / teacher-confidence attention / **跨相机错误使用 `(δu, δv)` 的对照实现**。
- [ ] **VLM 训练**：冻结 LLM / LoRA 8 / 16 / 32 / 仅 LM / LM + geometry retention / digit-wise waypoint / `<POS>` coordinate decoder。

验收：每个消融单一变量，adapter 容量保持一致；结果带 config hash。

---

### M15. 防捷径与因果验证（论文 §4.7，必跑）

任务（`shortcut_audit.py`）：
- [ ] **Image shuffle**：batch 内打乱 `image_embeds`，保留 LiDAR / 标定。
- [ ] **Calibration shuffle**：图像不变，交换 K / `T_ego→cam`。
- [ ] **Depth-label shuffle**：训练或评测打乱 `d_p`。
- [ ] **Allocation-label shuffle**：分别 shuffle `r_p`、`q_p^geom`、`a_p^*`。
- [ ] **Constant-depth baseline**：所有 token 使用固定深度或相机统计平均深度。
- [ ] **Relation-label shuffle**：保留深度，shuffle `cross_view` 与 `depth_order`。

预期表（写入论文 §4.7）：每项 shuffle 后必须出现性能下降；如果没有，立即回查模块。

验收：6 项实验脚本可一键执行（`shortcut_audit.sh`）；每项产出 JSON + 简表。

---

### M16. 可视化与定性分析

任务：
- [ ] 图 1 总体结构与训练 / 推理差异；
- [ ] 图 2 Vision Encoder + merger + 几何注入位置；
- [ ] 图 3 R²AC 风险 / 可靠度 / 灵敏度分配；
- [ ] 图 4 LiDAR privileged relation graph；
- [ ] 图 5 SpaceDrive 外部深度路径 vs LPGA；
- [ ] 图 6 spatial QA / 跨视角 grounding / 规划可视化；
- [ ] 数据卫生：跨视角标签同时报告正负边数量、可见性过滤比例、正边深度残差、人工抽样审计；
- [ ] 失败案例：spatial QA 与 planning 的 typical failure。

代码落点：`geodistill/utils/visualization.py`、`scripts/geodistill/visualize_sample.py`。

---

### M17. Paper-Ready Tables

任务：
- [ ] `scripts/geodistill/collect_tables.py`：从 eval JSON 汇总；
- [ ] 输出：
  - `runs/geodistill/tables/table1_spatial_qa.md`
  - `runs/geodistill/tables/table2_token_probe.md`
  - `runs/geodistill/tables/table3_relation_probe.md`
  - `runs/geodistill/tables/table4_open_loop.md`
  - `runs/geodistill/tables/table5_bench2drive.md`
  - `runs/geodistill/tables/table6_efficiency.md`
- [ ] 每个数字带 checkpoint / config hash / split id。

---

## 3. 推荐执行顺序（强约束）

```text
M0 (Qwen frozen forward) ──► M1 (dataset) ──► M2 (token-level LiDAR)
                                                      │
                                                      ▼
                                       (5 类 sanity check 必须通过)
                                                      │
                            ┌─────────────────────────┼─────────────────────────┐
                            ▼                         ▼                         ▼
              M3 (R²AC + risk)         M4 (LPGA encoder & heads)       M11 (driving QA 构造)
                            │                         │
                            └────────► M5 (R²AC loss + coord loss) ◄──┘
                                                      │
                                                      ▼
                                                 M7 (Stage A0)
                                                      │
                                                  (校准达标)
                                                      │
                                                      ▼
                                                 M8 (Stage A1)
                                                      │
                                                      ▼
                                       M6 (relation adapter + injection)
                                                      │
                                                      ▼
                                                 M9 (Stage B)
                                                      │
                                                      ▼
                                       M10 (Stage C, optional but recommended)
                                                      │
                                                      ▼
                       ┌──────────────────────────────┼──────────────────────────────┐
                       ▼                              ▼                              ▼
                  M12 评测套件               M13 / M14 baselines + 消融         M15 shortcut audit
                       └──────────────────────────────┬──────────────────────────────┘
                                                      ▼
                                              M16 可视化 + M17 paper tables
```

硬门槛：
- M2 没通过 5 类 sanity check，不进入 M5 / M6 训练；
- A0 校准（`r_hat/q_hat/a_hat` Spearman）没达标，不进入 A1；
- A1 token AbsRel 没达标，不进入 B；
- B 第 0 步若 `H_geo ≠ H_img`，必须先修零初始化，再训练；
- M15 shortcut audit 任一项未出现性能下降，对应 contribution 不进入论文。

---

## 4. MVP（最小可投稿闭环）

### MVP-1：Qwen frozen + token label cache（M0–M2）
- 单 sample 跑通 `image_embeds`；
- mini split token 级 LiDAR 标签缓存；
- 5 类 sanity check 全通过。

### MVP-2：R²AC + LPGA depth-only（M3–M5 + M7 + M8）
- A0 → A1 在 mini split 上完成；
- 报告 token AbsRel、`r̂/q̂/â` 校准、risk-stratified AbsRel。

### MVP-3：几何注入 + LoRA spatial QA（M6 + M9 + M11 + M12 部分）
- 只跑 spatial QA + token probe + relation probe，先证明几何写回 LLM 有正向收益。

### MVP-4：闭环 + 鲁棒 + shortcut audit（M10 + M15 + M12 余下）
- nuScenes 开环 + Bench2Drive 闭环主表；
- 鲁棒性表；
- 6 项 shortcut shuffle 全部出现性能下降。

MVP 成功标准：
- 表 2 token probe 优于 SpaceDrive-style 与 LiDAR-calibrated SpaceDrive 在 30+ 段位的退化；
- 表 1 spatial QA hard paraphrase 优于 LoRA-only；
- 表 5 Bench2Drive 主协议优于 LoRA-only 且不劣于 LiDAR-calibrated SpaceDrive；
- 表 6 形成精度—延迟 Pareto；
- shortcut audit 6 项全部出现下降。

---

## 5. 风险与处理

### R1. Qwen2.5-VL 7B 显存压力
处理：
- 默认 `640×640` + 6 cam，按 SpaceDrive 配置；
- 阶段 A0/A1 关闭 LoRA 与 `W_up`，只训练 LPGA；
- 阶段 B LoRA rank-16，启用 gradient checkpointing。

### R2. LiDAR token-level 标签噪声
处理：
- foreground quantile + centroid + 多帧 ego-motion + dynamic mask；
- `q_p^label = s_p · q_p^geom` 抑制单点 token；
- `L_topo / L_occ` 默认关闭，5 类 sanity check 通过后再开。

### R3. Allocation 漂移与退化解
处理：
- 始终 `u_p^* = F(d_p, a_p^*)` 用 oracle；
- `d_hat = F_inv(u_hat, StopGrad(a_hat))`；
- A0 warm-up 在 metric 解码与 coord loss 启用前稳定 `r_hat/q_hat`；
- 主路径不叠加 `L_a^aux`。

### R4. 几何注入破坏 Qwen 语义
处理：
- 双 gate 严格零初始化 + `tanh(β)`；
- `W_up` 小方差非零初始化避免梯度消失；
- 阶段 B 启用 `L_keep` 限制全局语义漂移。

### R5. 跨相机关系误用 `(δu, δv)`
处理：
- 同视角与跨相机分支独立 edge encoder；
- 构造期 assert 保证特征字段；
- 提供 §4.6 「跨相机错误使用 `δu/δv`」对照实验。

### R6. 数据集与 baseline 复现
处理：
- SpaceDrive-style 用冻结 UniDepthV2 + 3D PE，照搬 §4.4 表述；
- LiDAR-calibrated SpaceDrive 与 LPGA 共用 LiDAR depth/coord 监督预算；
- DriveLM / OmniDrive QA 不能与 hard paraphrase 模板共享。

### R7. Bench2Drive vs Bench2Drive-VL 协议泄漏
处理：
- 主协议禁止 Bench2Drive-VL 数据；
- 如做增强训练，必须独立行；
- 收敛标准与 closed-loop 评测必须使用同一路由 / 交通密度 / 传感器配置。

### R8. 因果验证不通过
处理：
- 立即回查 Token encoder 输入隔离；
- 回查 calibration / ego state 是否被错误注入；
- 必要时撤回对应 contribution。

### R9. 第三方代码（SpaceDrive / UniDepth）边界
处理：
- 以 git submodule 引入，pin commit hash，禁止 fork-and-modify；
- 不在主路径 `geodistill/models/*` 内 import SpaceDrive；只在 `geodistill/baselines/` 与 evaluation tooling 调用；
- `pe_3d.py` / `coord_decoder.py` 自研，避免与 SpaceDrive 共享实现而稀释 contribution；
- `THIRD_PARTY_LICENSES.md` 列出 MIT 来源、commit、是否修改、修改清单；
- 如本地无法复现 SpaceDrive 训练，回退到使用其官方 HF checkpoint（`spacedrive_qwen` / `spacedrive_llava`）评测，并在论文标注「pretrained ckpt only」。

---

## 6. 当前下一步

1. **M0a**：`git submodule add third_party/SpaceDrive`，写 `THIRD_PARTY_LICENSES.md`；
2. **M0b**：搭起 `geodistill/` + 配置 + Qwen2.5-VL frozen forward；
3. M1：基于现有 `dataset/geotoken/nuscenes_clip_dataset.py` 扩展 Qwen processor 与 token meta；
4. M2：token-level LiDAR label builder + 5 类 sanity check（这是第一阶段最大风险点，必须人工审 50 个 token）；
5. M3 + M4：R²AC 单元测试 + LPGA token encoder & heads（含输入隔离 assert）；
6. M5：R²AC loss + coord loss，跑通单 batch overfit；
7. M7 / M8：Stage A0 → A1 mini split；
8. 与 M7/M8 并行启动 **M13 baseline 3 / 4**（直接调用 `third_party/SpaceDrive` 跑 SpaceDrive-style 与 LiDAR-calibrated 版本），保证主结果发布时 baseline 可比；
9. 在 spatial QA 介入前，必须先有 token probe + relation probe 表 2 / 表 3；
10. 之后才允许进入 M6 几何注入与 M9 阶段 B。


