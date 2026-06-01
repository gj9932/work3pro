# work3 Pro 任务拆分：GeoToken / PSRD / LGSS

> 分支：`work3pro-geotoken`
>
> 日期：2026-05-31
>
> 目标：完全转向 driving VLM spatial reasoning，按 `plan/work3pro.md` 和 `paper/work3pro_cvpr2027_draft3.md` 重构为：
>
> **LiDAR-privileged relation pretraining for language-accessible camera-only spatial tokens in autonomous driving.**

---

## 0. 总体边界

### 0.1 新工作核心

新工作围绕三件事展开：

1. **GeoToken tokenizer**
   - 输入：nuScenes 多视角相机 clip。
   - 输出：BEV anchor tokens、spatial slots、VLM adapter tokens。
   - 推理：只使用 camera，不使用 LiDAR。

2. **PSRD: Privileged Spatial Relation Distillation**
   - 训练阶段使用对齐的 LiDAR、六视角 camera、ego pose、calibration 生成 ego-centric relation graph。
   - 3D box / map metadata 只作为可选标签来源，用于 phrase、QA 构造和评测，不是部署输入。
   - 监督 camera BEV anchor token 的 metric、direction、topology、occlusion relations。

3. **LGSS: Language-Grounded Spatial Slots**
   - 从 BEV anchor tokens 读出 object / region / relation slots。
   - 用弱空间短语和 spatial QA head 让 frozen VLM 可以访问 token 内的空间结构。

### 0.2 不再作为主约束的内容

以下内容不作为新分支的主约束：

- 不要求复用原 `SPR/TPR + NetVLAD + GRL + multi-manifold distillation` 训练框架。
- 不再设计地点识别、descriptor distillation、localization 相关任务。
- 不再把原工作作为实验对照；旧代码只作为数据读取或工程参考。

原代码可以作为参考或迁移来源：

- `dataset/NuScenesDataset.py`：多相机和 range data 读取逻辑。
- `tools/gen_info.py`：nuScenes info 生成逻辑。
- `tools/gen_range.py`：LiDAR range projection 参考。
- `tools/runner.py`：训练/验证脚手架参考。
- `modules/S_net.py`、`modules/T_net.py`：旧模型文件，仅作为工程参考，不作为新方法 baseline。

---

## 1. 目标目录规划

建议新增目录，不在旧模块上硬改。

```text
configs/geotoken/
  geotoken_nuscenes.yaml
  ablation_psrd.yaml
  adapter_vlm.yaml

dataset/geotoken/
  nuscenes_clip_dataset.py
  nuscenes_geometry.py
  spatial_qa_dataset.py
  corruptions.py

geotoken/
  models/
    camera_tokenizer.py
    bev_anchor_queries.py
    view_fusion.py
    temporal_encoder.py
    spatial_slots.py
    vlm_adapter.py
  geometry/
    bev_grid.py
    lidar_rasterizer.py
    projection.py
    relation_graph.py
    edge_sampler.py
    phrase_generator.py
  losses/
    psrd_loss.py
    lgss_loss.py
    auxiliary_losses.py
  trainers/
    train_psrd.py
    train_lgss.py
    train_vlm_adapter.py
  eval/
    eval_spatial_qa.py
    eval_spatial_probe.py
    eval_robustness.py
  utils/
    distributed.py
    checkpoint.py
    logging.py
    visualization.py

scripts/geotoken/
  build_nuscenes_infos.py
  build_relation_cache.py
  build_spatial_qa.py
  train_psrd.sh
  train_lgss.sh
  train_adapter.sh
  eval_all.sh
```

---

## 2. Milestone 拆分

### M0. 仓库重构和实验配置

目标：建立新分支的 GeoToken 工程骨架，面向 VLM spatial reasoning 重新组织代码。

任务：

- [ ] 新建 `configs/geotoken/` 配置目录。
- [ ] 新建 `dataset/geotoken/` 数据目录。
- [ ] 新建 `geotoken/` 主包。
- [ ] 新建 `scripts/geotoken/` 脚本目录。
- [ ] 保留旧 `modules/`、`tools/`，但新代码不依赖旧 trainer。
- [ ] 增加统一配置字段：
  - dataset root；
  - camera names；
  - clip length `T`；
  - BEV range；
  - grid size；
  - active anchor count；
  - edge sampling count；
  - PSRD loss weights；
  - LGSS slot count；
  - VLM adapter type。

输出：

- `configs/geotoken/geotoken_nuscenes.yaml`
- 新代码目录骨架。

验收：

- 可以 import `geotoken` 包。
- 配置能被脚本读取并打印完整实验参数。

---

### M1. nuScenes 多视角 clip 数据层

目标：构建多视角、多帧、带几何 metadata 的 camera-only 输入。

任务：

- [ ] 实现 `NuScenesClipDataset`。
- [ ] 每个样本返回：
  - `images`: `[T, N_cam, 3, H, W]`
  - `camera_intrinsics`
  - `camera_extrinsics`
  - `ego_poses`
  - `sample_tokens`
  - `scene_token`
  - `timestamp`
  - 训练阶段可选：`lidar_path`、`boxes_3d`、`map_labels`
- [ ] 支持 `T=1/3/5`。
- [ ] 支持 center frame 对齐。
- [ ] 支持 six-camera 默认顺序：
  - `CAM_FRONT`
  - `CAM_FRONT_RIGHT`
  - `CAM_BACK_RIGHT`
  - `CAM_BACK`
  - `CAM_BACK_LEFT`
  - `CAM_FRONT_LEFT`
- [ ] 实现 camera corruption hook：
  - single camera drop；
  - multi-camera drop；
  - front-only；
  - limited FOV；
  - random image occlusion；
  - blur / rain / night style corruption。

代码落点：

- `dataset/geotoken/nuscenes_clip_dataset.py`
- `dataset/geotoken/corruptions.py`
- `scripts/geotoken/build_nuscenes_infos.py`

输出：

- 可用于 PSRD 训练的 dataloader。
- 可用于 robustness evaluation 的 corruption dataloader。

验收：

- 单 batch shape 正确。
- 六相机图像顺序稳定。
- calibration 和 ego pose 可追踪到 center frame。
- corruption 不改变 batch schema。

---

### M2. Ego-Centric BEV Anchor Grid

目标：定义 PSRD 的唯一主图节点类型：fixed ego-centric BEV anchors。

任务：

- [ ] 实现 BEV grid builder。
- [ ] 默认范围：
  - `x in [-40m, 40m]`
  - `y in [-40m, 40m]`
  - cell size `2m x 2m`
  - candidate anchors `40 x 40 = 1600`
- [ ] 每个 anchor 保存：
  - anchor id；
  - ego-frame center `(x, y)`；
  - cell bounds；
  - camera frustum visibility；
  - optional occupancy/free-space labels。
- [ ] 实现 active anchor selector。
- [ ] 训练默认选择 `L=400` active anchors，优先级：
  - 3D box covered anchors；
  - occupied LiDAR anchors；
  - drivable/free-space boundary anchors；
  - camera-frustum visible anchors；
  - uniform free-space anchors。
- [ ] 推理阶段实现两种模式：
  - full 1600 anchors with chunked attention；
  - camera-only prior top-400 anchors。

代码落点：

- `geotoken/geometry/bev_grid.py`
- `geotoken/models/bev_anchor_queries.py`

输出：

- `anchors_candidate`: `[1600, 2]`
- `anchors_active`: `[400, 2]`
- `anchor_masks`

验收：

- active anchor 选择不在推理依赖 LiDAR / GT box。
- object / image patch 不作为 PSRD 主图节点。
- anchor 坐标和论文定义一致。

---

### M3. LiDAR 几何 teacher 和 Relation Cache

目标：把 LiDAR、ego pose、calibration 转成 PSRD 训练标签。

任务：

- [ ] 读取 center frame LiDAR 点云。
- [ ] 将 LiDAR 点转换到 ego coordinate。
- [ ] rasterize 到 BEV anchors。
- [ ] 生成 occupancy / free-space / unknown labels。
- [ ] 可选读取 nuScenes 3D boxes，用于：
  - object covered anchors；
  - object-level pooling label；
  - weak spatial phrase generation。
- [ ] 实现 camera projection：
  - anchor center 到各 camera image plane；
  - camera frustum visibility；
  - ray neighborhood index；
  - depth ordering。
- [ ] 实现 relation graph labels：
  - distance bins；
  - direction bins；
  - topology classes；
  - occlusion classes；
  - reliability mask `m_ij`；
  - occlusion mask `m_occ_ij`。
- [ ] 实现 relation cache 预生成脚本，避免每轮重复计算几何。

Relation label 默认定义：

```text
distance bins:
  [0,2), [2,5), [5,10), [10,20), [20,40), >=40m

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

代码落点：

- `geotoken/geometry/lidar_rasterizer.py`
- `geotoken/geometry/projection.py`
- `geotoken/geometry/relation_graph.py`
- `scripts/geotoken/build_relation_cache.py`

输出：

- 每个 sample 的 relation cache：
  - active anchor ids；
  - anchor labels；
  - sampled edge labels；
  - masks；
  - phrase source metadata。

验收：

- 可视化 BEV occupancy 与 LiDAR 点云对齐。
- 可视化 camera projection 在图像内位置合理。
- occlusion label 只在可靠 ray neighborhood 内产生。
- `m_ij` 和 `m_occ_ij` 不把低质量 pair 混入 loss。

---

### M4. Sparse Edge Sampler

目标：把 PSRD 从 `O(L^2)` 控制到 `O(LP)`。

任务：

- [ ] 实现 per-anchor edge sampling。
- [ ] 默认每个 anchor 采样 `P=32` 条边。
- [ ] 边类型：
  - `P_near`: metric nearest valid anchors；
  - `P_topo`: same free-space / same occupied component；
  - `P_occ`: valid occlusion-order pairs；
  - `P_hard`: different topology component but similar visual prior；
  - `P_rand`: random long-range valid anchors。
- [ ] 保存 edge type，用于 loss 分析。
- [ ] 支持 dense edge mode，仅用于 probe / visualization。

代码落点：

- `geotoken/geometry/edge_sampler.py`

输出：

- `edge_index`: `[E, 2]`
- `edge_labels`
- `edge_masks`
- `edge_type`

验收：

- `L=400, P=32` 时每帧边数不超过 `12.8k`。
- 每类 edge 在训练集上有非零覆盖。
- hard negative 不依赖测试标签。

---

### M5. Camera Spatial Tokenizer

目标：实现 GeoToken 的 camera-only token 生成器。

任务：

- [ ] 实现 image frame encoder。
  - 第一阶段可用 DINOv2 / SigLIP / CLIP-ViT。
  - 如果本地依赖不可用，先实现 backbone interface 和 ResNet fallback。
- [ ] 实现 camera-id embedding。
- [ ] 实现 view-aware fusion。
- [ ] 实现 calibration-aware BEV anchor cross-attention。
- [ ] 实现 temporal encoder。
- [ ] 输出：
  - BEV anchor tokens `Z`；
  - intermediate patch tokens；
  - camera-only anchor priors。
- [ ] 支持训练阶段 active 400 anchors。
- [ ] 支持推理阶段 full 1600 anchors 或 predicted top-400 anchors。

代码落点：

- `geotoken/models/camera_tokenizer.py`
- `geotoken/models/view_fusion.py`
- `geotoken/models/temporal_encoder.py`
- `geotoken/models/bev_anchor_queries.py`

输出：

- `Z`: `[B, L, C]`
- `anchor_prior`: `[B, 1600]`

验收：

- 不输入 LiDAR 时可正常 forward。
- `Z` 和 active anchors 对齐。
- multi-view 和 temporal 输入 shape 稳定。

---

### M6. PSRD Loss

目标：实现论文主损失，直接监督 camera token 的空间关系结构。

任务：

- [ ] 实现 camera relation distribution：

```text
R_cam_ij = softmax_j((W_q z_i)^T (W_k z_j) / sqrt(C))
```

- [ ] 实现 LiDAR relation distribution：

```text
R_lidar_ij =
  exp(-alpha d_ij) * 1[m_ij = 1] * rho(o_ij)
  ------------------------------------------------
  sum_k exp(-alpha d_ik) * 1[m_ik = 1] * rho(o_ik)
```

- [ ] 实现 `rho(o_ij)`：

```text
1.5  same-free-space
1.3  same-occupied-component
0.8  free-to-occupied / occupied-to-free
0.5  different-free-components
0.0  unknown
```

- [ ] 实现 pairwise heads：
  - distance；
  - direction；
  - topology；
  - occlusion。
- [ ] 实现 losses：

```text
L_psrd =
  lambda_rel  L_rel
+ lambda_dist L_dist
+ lambda_dir  L_dir
+ lambda_topo L_topo
+ lambda_occ  L_occ
```

- [ ] 默认权重：
  - `lambda_rel=1.0`
  - `lambda_dist=1.0`
  - `lambda_dir=0.5`
  - `lambda_topo=0.5`
  - `lambda_occ=0.5`
- [ ] 支持 topology / occlusion inverse-frequency class weights。

代码落点：

- `geotoken/losses/psrd_loss.py`

输出：

- loss dict：
  - `loss_psrd`
  - `loss_rel`
  - `loss_dist`
  - `loss_dir`
  - `loss_topo`
  - `loss_occ`
  - relation accuracy metrics。

验收：

- mask 生效，invalid edge 不进入 loss。
- occlusion loss 只使用 `m_occ_ij=1` 的 pair。
- loss 在 toy batch 上可反向传播。

---

### M7. PSRD Pretraining Runner

目标：训练 camera tokenizer，使 camera-only tokens 学到 LiDAR privileged relations。

任务：

- [ ] 实现 `train_psrd.py`。
- [ ] 支持：
  - resume；
  - checkpoint；
  - tensorboard / wandb optional logging；
  - AMP；
  - gradient clipping；
  - multi-GPU optional。
- [ ] 总损失第一阶段只启用：

```text
L_total = L_psrd
```

- [ ] 第二阶段加入辅助项：
  - temporal ego-motion consistency；
  - sensor degradation consistency；
- [ ] 每个 epoch 输出：
  - PSRD loss；
  - pairwise relation accuracy；
  - valid edge ratio；
  - active anchor coverage；
  - occlusion pair coverage。

代码落点：

- `geotoken/trainers/train_psrd.py`
- `geotoken/losses/auxiliary_losses.py`
- `scripts/geotoken/train_psrd.sh`

输出：

- `checkpoints/geotoken_psrd_*.pth`

验收：

- 单 batch overfit loss 可下降。
- 小规模 train split 可跑完整 epoch。
- checkpoint 可恢复并继续训练。

---

### M8. Frozen Spatial Probing

目标：在不接 VLM 前，先证明 token 本身有空间结构。

任务：

- [ ] 冻结 camera tokenizer。
- [ ] 训练小 probing heads：
  - depth bin / depth regression；
  - BEV occupancy；
  - distance relation；
  - direction relation；
  - topology relation；
  - occlusion relation。
- [ ] 指标：
  - depth AbsRel / RMSE / delta；
  - BEV IoU / mIoU；
  - relation accuracy；
  - occlusion accuracy。
- [ ] baselines：
  - raw DINOv2/SigLIP/CLIP patch tokens；
  - depth-only auxiliary；
  - BEV-only auxiliary；
  - GeoToken PSRD。

代码落点：

- `geotoken/eval/eval_spatial_probe.py`

输出：

- `runs/geotoken/probe_results.json`
- Table 2 数据。

验收：

- 所有 probing 都冻结 tokenizer。
- baseline 使用同等 probe capacity。
- 输出能直接填论文 Table 2。

---

### M9. LGSS Spatial Slots

目标：让 PSRD 学到的空间 token 可被语言接口访问。

任务：

- [ ] 实现 learnable slot reader：

```text
S = CrossAttn(Q_slot, Z)
```

- [ ] slot 分三类：
  - object slots；
  - region slots；
  - relation slots。
- [ ] 实现 slot attention map `A`。
- [ ] 实现 Hungarian matching。
- [ ] matching cost：

```text
cost(s_k, y_j) =
  lambda_iou  (1 - IoU(A_k, A_j))
+ lambda_cls  CE(c_hat_k, c_j)
+ lambda_txt  (1 - sim(s_k, t_j))
```

- [ ] 实现 phrase embedding interface：
  - CLIP-aligned VLM 使用 CLIP text encoder；
  - Qwen-VL / InternVL / LLaVA-style 使用目标 VLM tokenizer + language embedding stack + projection；
  - CLIP text encoder 作为 controlled ablation。
- [ ] 实现 multi-positive contrastive slot loss。
- [ ] 实现 pairwise QA head：
  - closer/farther；
  - left/right/front；
  - free/occupied；
  - occluded/not occluded；
  - cross-view same object。
- [ ] 实现 diversity loss：

```text
L_div = || A A^T - I ||_F
```

- [ ] 实现 temporal slot consistency：

```text
L_slot_temp = || Warp(A_t, T_t->t+1) - A_t+1 ||_1
```

代码落点：

- `geotoken/models/spatial_slots.py`
- `geotoken/losses/lgss_loss.py`
- `geotoken/geometry/phrase_generator.py`
- `geotoken/trainers/train_lgss.py`

输出：

- `spatial_slots`: `[B, K, C]`
- slot labels / phrase labels。
- LGSS checkpoint。

验收：

- object / region / relation slots 不共享同一组 query embedding。
- slot attention 可视化能定位到合理 anchor 区域。
- no-diversity ablation 可关闭。

---

### M10. Spatial QA 数据构造

目标：构造真实 driving spatial QA，而不是只做特征 probe。

任务：

- [ ] 从 nuScenes metadata / 3D boxes / LiDAR relation graph 生成 QA。
- [ ] QA 类别：
  - distance；
  - direction；
  - topology/free-space；
  - occlusion；
  - cross-view；
  - temporal motion。
- [ ] 每条 QA 保存：
  - question；
  - answer；
  - question type；
  - involved object ids / anchor ids；
  - source relation；
  - reliability score；
  - split id。
- [ ] 实现 hard paraphrase split：
  - train phrases 和 test questions 使用不同表达。
- [ ] 实现 object-composition split：
  - train/test 使用不同 object-relation 组合。
- [ ] 实现 template split：
  - 同 relation category，不同 scene instance。
- [ ] 过滤低可靠 QA：
  - sparse LiDAR；
  - ambiguous occlusion；
  - unstable dynamic object association；
  - object too small / too far。

代码落点：

- `dataset/geotoken/spatial_qa_dataset.py`
- `geotoken/geometry/phrase_generator.py`
- `scripts/geotoken/build_spatial_qa.py`

输出：

- `spatial_qa_train.jsonl`
- `spatial_qa_val.jsonl`
- `spatial_qa_test_template.jsonl`
- `spatial_qa_test_paraphrase.jsonl`
- `spatial_qa_test_composition.jsonl`

验收：

- 每类 QA 数量统计完整。
- hard paraphrase split 无模板泄漏。
- QA answer 可从 relation graph 复核。

---

### M11. Frozen VLM Adapter

目标：证明 GeoToken 对 frozen driving VLM 有用，而不是只证明 token probe 好。

任务：

- [ ] 选择首个 frozen VLM。
  - 推荐优先从 Qwen-VL / InternVL / LLaVA-Video 选一个本地可跑模型。
  - 如果本地算力不足，先实现 VLM adapter interface 和 small text decoder proxy。
- [ ] 冻结：
  - camera tokenizer；
  - LGSS；
  - VLM。
- [ ] 只训练 adapter：

```text
U = A_psi([Z, S])
```

- [ ] adapter 类型：
  - MLP projector；
  - Q-Former-style adapter。
- [ ] 统一不同 baseline 的 adapter capacity。
- [ ] 输出 VLM answers。

代码落点：

- `geotoken/models/vlm_adapter.py`
- `geotoken/trainers/train_vlm_adapter.py`
- `geotoken/eval/eval_spatial_qa.py`
- `scripts/geotoken/train_adapter.sh`

输出：

- adapter checkpoint。
- Table 1 spatial QA 结果。

验收：

- tokenizer 和 VLM 参数冻结。
- GeoToken、DINOv2 baseline、depth baseline 使用相同 adapter 容量。
- 结果按 category 和 hard paraphrase 分开统计。

---

### M12. Robustness Evaluation

目标：把 robustness 作为主实验之一。

任务：

- [ ] 在 clean / corrupt 输入上评估 spatial QA。
- [ ] corruption types：
  - single camera drop；
  - multiple camera drop；
  - front-only camera；
  - limited FOV；
  - random image occlusion；
  - night；
  - rain；
  - motion blur。
- [ ] 指标：

```text
Robustness Ratio = Performance_corrupt / Performance_clean
```

- [ ] 输出：
  - clean QA；
  - robust QA；
  - QA ratio；
  - category-wise ratio。

代码落点：

- `dataset/geotoken/corruptions.py`
- `geotoken/eval/eval_robustness.py`

输出：

- Table 3 robustness 结果。

验收：

- corruption 对所有方法一致。
- 每个 corruption 单独报告，再汇总平均。
- front-only 不使用其他 camera 信息。

---

### M13. Ablation Study

目标：回答 reviewer 对方法来源的质疑。

任务：

- [ ] PSRD 组件消融：
  - no LiDAR privileged supervision；
  - depth-only；
  - BEV-only；
  - no metric distance；
  - no direction；
  - no topology；
  - no occlusion order。
- [ ] LGSS 消融：
  - no LGSS；
  - no diversity；
  - no temporal slot consistency；
  - CLIP text encoder；
  - mismatched text encoder；
  - target-VLM text space；
  - shuffled phrases。
- [ ] 架构输入消融：
  - `T=1/3/5`；
  - front-only / 3 cameras / 6 cameras；
  - MLP adapter / Q-Former adapter；
  - active 400 / full 1600 anchors。

代码落点：

- `configs/geotoken/ablation_psrd.yaml`
- `geotoken/eval/eval_spatial_qa.py`
- `geotoken/eval/eval_spatial_probe.py`

输出：

- Table 4 / Table 5 数据。

验收：

- 每个消融只改一个核心变量。
- adapter capacity 保持一致。
- 结果文件包含 config hash。

---

### M14. Visualization 和 Qualitative Analysis

目标：为论文图和 debug 提供可解释可视化。

任务：

- [ ] Figure 1：GeoToken training/inference overview。
- [ ] Figure 2：LiDAR ego-centric relation graph construction。
- [ ] Figure 3：PSRD relation / distance / direction / topology / occlusion losses。
- [ ] Figure 4：LGSS + frozen VLM adapter protocol。
- [ ] Figure 5：clean vs corrupted token affinity / slot attention。
- [ ] 可视化工具：
  - BEV anchors；
  - active anchor subset；
  - occupancy/free-space；
  - sampled edges；
  - camera projection；
  - slot attention；
  - token affinity map。

代码落点：

- `geotoken/utils/visualization.py`
- `scripts/geotoken/visualize_sample.py`

输出：

- `runs/geotoken/vis/*.png`

验收：

- 每张图能从真实 sample 自动生成。
- 图中能区分 LiDAR teacher graph 和 camera token graph。
- qualitative failure case 有标注。

---

### M15. Paper-Ready Experiment Tables

目标：把工程输出直接对齐论文表格。

任务：

- [ ] Table 1：Driving spatial QA。
- [ ] Table 2：Frozen spatial probing。
- [ ] Table 3：Robustness under camera degradation。
- [ ] Table 4：PSRD and LGSS ablation。
- [ ] Table 5：Architecture and input ablation。
- [ ] 每个 evaluation script 输出统一 JSON：

```json
{
  "method": "...",
  "config": "...",
  "checkpoint": "...",
  "split": "...",
  "metrics": {}
}
```

- [ ] 增加 `scripts/geotoken/collect_tables.py` 汇总为 markdown / csv。

代码落点：

- `scripts/geotoken/collect_tables.py`

输出：

- `runs/geotoken/tables/table1_spatial_qa.md`
- `runs/geotoken/tables/table2_probe.md`
- `runs/geotoken/tables/table3_robustness.md`
- `runs/geotoken/tables/table4_ablation.md`
- `runs/geotoken/tables/table5_arch.md`

验收：

- 表格列名和论文 draft 一致。
- 每个数字能追踪到 checkpoint 和 config。

---

## 3. 推荐执行顺序

### Phase A：几何数据闭环

优先级最高。没有几何标签，PSRD 无法成立。

1. M0 仓库骨架。
2. M1 nuScenes clip dataset。
3. M2 BEV anchor grid。
4. M3 LiDAR relation cache。
5. M4 sparse edge sampler。

阶段验收：

- 随机 sample 可生成 active anchors、relation graph、edge labels。
- 可视化 LiDAR BEV、camera projection、edge relation。

### Phase B：PSRD token pretraining

1. M5 camera spatial tokenizer。
2. M6 PSRD loss。
3. M7 PSRD pretraining runner。
4. M8 frozen spatial probing。

阶段验收：

- relation probe 比 raw image token 更好。
- occlusion probe 在启用 `L_occ` 后有提升。
- topology probe 能区分 free / occupied / different component。

### Phase C：语言可访问和 VLM

1. M9 LGSS。
2. M10 spatial QA 构造。
3. M11 frozen VLM adapter。

阶段验收：

- hard paraphrase split 上 GeoToken 优于同容量 baseline。
- no-LGSS 低于 full GeoToken。
- shuffled phrases 明显下降，证明不是模板泄漏。

### Phase D：鲁棒性和论文实验

1. M12 robustness。
2. M13 ablation。
3. M14 visualization。
4. M15 paper-ready tables。

阶段验收：

- clean 和 corrupt 都有完整结果。
- 消融能支撑论文 claim。
- 可视化能解释 token 学到的 relation。

---

## 4. 最小可行版本 MVP

如果先做一个能出初步结果的 MVP，范围如下：

### MVP-1：Relation Cache

- 单帧 `T=1`。
- 六相机。
- 400 active anchors。
- distance / direction / topology 三类 relation。
- occlusion 先实现 mask 和接口，随后补 loss。

### MVP-2：Camera Tokenizer + PSRD

- DINOv2 或 ResNet fallback。
- view-aware fusion。
- BEV anchor cross-attention。
- `L_rel + L_dist + L_dir + L_topo`。

### MVP-3：Frozen Probe

- relation accuracy。
- BEV occupancy IoU。
- depth bin accuracy。

### MVP-4：简版 QA

- 不先接大 VLM。
- 用 frozen tokenizer + small adapter / text proxy 做 spatial QA。
- QA 类别先做 distance、direction、free-space。

MVP 成功标准：

- PSRD token 在 relation probe 上超过 raw image token。
- BEV occupancy probing 有稳定提升。
- 简版 spatial QA 上 GeoToken 优于 DINOv2 adapter baseline。

---

## 5. 风险和处理

### R1. LiDAR relation label 噪声大

处理：

- 强化 `m_ij` 和 `m_occ_ij`。
- 先做 distance/direction/topology，occlusion 后置。
- 只在可视化验证可靠后启用 occlusion loss。

### R2. VLM 接入成本过高

处理：

- 先实现 VLM adapter interface。
- 用 small text decoder proxy 做 MVP。
- 主论文最终必须至少接一个 frozen driving/general VLM。

### R3. LGSS 被质疑为模板记忆

处理：

- hard paraphrase split。
- object-composition split。
- shuffled phrases ablation。
- no-LGSS ablation。
- target-VLM text space vs CLIP text encoder ablation。

### R4. 算力压力

处理：

- 默认 active 400 anchors。
- sparse `P=32` edges。
- full 1600 anchors 只用于 inference / ablation。
- 第一阶段 `T=1`，之后扩展 `T=3/5`。

### R5. 原代码迁移成本

处理：

- 新代码独立实现。
- 旧代码只作为数据读取、range projection、baseline 参考。
- 不在旧 trainer 上堆新功能。

---

## 6. 当前下一步

建议马上执行：

1. 创建 `configs/geotoken/`、`dataset/geotoken/`、`geotoken/`、`scripts/geotoken/`。
2. 实现 `NuScenesClipDataset` 的 batch schema。
3. 实现 BEV anchor grid 和可视化。
4. 实现 LiDAR relation cache 的最小版本。
5. 用 10 个 sample 做 relation graph sanity check。

完成上述 5 步后，再进入 camera tokenizer 和 PSRD loss。
