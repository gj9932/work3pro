# work3 → VLM / 视频理解升级计划 (Living Document)

> 本文件是 `work3plan1.md` 的方向升级版，重点记录如何把 work3 从“跨模态视觉位置识别”升级为 2026 热点方向：**自动驾驶 VLM / 视频理解 / 多模态时空表征学习**。
> 每个新会话开始前先读本文件；若继续做代码或论文改造，结束前必须更新 `## 10. 会话日志`。

---

## 0. 核心判断

如果 work3 后续想为 VLM、视频理解、自动驾驶场景问答、驾驶行为解释做铺垫，**不建议继续把它只写成 VPR 论文**。

更合适的定位是：

> **LiDAR-privileged geometry-aware visual/video pretraining for robust driving scene understanding**

中文表述：

> 利用训练阶段可获得的 LiDAR 几何信息，监督相机视频模型学习具备三维结构感知能力的时空视觉表征，使其在推理阶段仅依赖相机，也能支持鲁棒的地点识别、视频理解和后续自动驾驶 VLM 任务。

这条路线的优势：
- 保留 work3 原有资产：LiDAR teacher、camera student、多流形蒸馏、nuScenes 数据。
- 顺应 2026 热点：VLM、video understanding、world model、camera-only driving、sensor-failure robustness。
- 比单纯“ResNet → DINOv2”更有论文价值。
- 后续可以自然接 Qwen-VL / InternVL / LLaVA-OneVision / DriveVLM 类系统。

---

## 1. 推荐新题目方向

### 首选题目

**GeoDrive: LiDAR-Privileged Geometry-Aware Visual Pretraining for Robust Driving Video Understanding**

### 备选题目

- **GeoVLM-Drive: Geometry-Aware Cross-Modal Pretraining for Driving Vision-Language Models**
- **DriveGeoDistill: LiDAR-Guided Video Representation Learning for Camera-Only Autonomous Driving**
- **GeoVPR-Video: Sensor-Failure-Robust Spatiotemporal Place Understanding with LiDAR-Privileged Distillation**
- **GCD-Drive: Geometry-aware Cross-modal Distillation for Driving Video Understanding**

### 不建议继续使用的弱题目

- “多流形蒸馏的视觉位置识别”
- “跨模态领域自适应视觉位置识别”
- “LiDAR-to-camera distillation for VPR”

原因：这些表述显得窄，像 2023-2024 的 VPR / KD 工作，不容易体现 2026 的热点趋势。

---

## 2. 新 story

### 原 work3 story

> 训练阶段使用 LiDAR 和 camera，通过多流形蒸馏和对抗学习，把 LiDAR 知识迁移到 camera student；推理阶段只用 camera 做视觉位置识别。

### 升级后 story

> We treat LiDAR as a privileged geometric teacher available only during training. Through geometry-aware cross-modal distillation, the camera/video student learns spatial, temporal, and global scene representations that are robust to sensor degradation. The learned tokens can serve both traditional place recognition and downstream driving VLM/video-understanding tasks.

中文：

> 我们将 LiDAR 视为训练阶段可用的特权几何教师，通过几何感知跨模态蒸馏，使相机视频学生模型学习空间、时间和全局场景表征。该表征不仅支持相机-only 位置识别，也可作为后续自动驾驶 VLM 和视频理解任务的几何感知视觉 token。

---

## 3. 两篇最重要对标论文

### 3.1 Sensor-failure robustness 对标

**Resilient Sensor Fusion Under Adverse Sensor Failures via Multi-Modal Expert Fusion**  
CVPR 2025  
链接：`https://openaccess.thecvf.com/content/CVPR2025/html/Park_Resilient_Sensor_Fusion_Under_Adverse_Sensor_Failures_via_Multi-Modal_Expert_CVPR_2025_paper.html`

相关点：
- 明确研究 LiDAR-camera 系统在 sensor failure 下的鲁棒性。
- failure 包括：LiDAR beam reduction、LiDAR drop、limited FOV、camera drop、occlusion。
- 使用 `nuScenes-R / nuScenes-C` 风格评测。
- 说明“传感器降级容错”是 CVPR 级别认可的问题。

work3 应借鉴：
- 将“训练用 LiDAR、推理只用 camera”包装成 **sensor degradation / sensor-failure robustness**。
- 加 camera drop、view drop、limited-FOV、night/rain 子集评测。
- 不一定做 object detection，但要把 VPR / video understanding 任务也放进 failure protocol。

### 3.2 VLM / camera-only driving 对标

**SimLingo: Vision-Only Closed-Loop Autonomous Driving with Language-Action Alignment**  
CVPR 2025  
链接：`https://cvpr.thecvf.com/virtual/2025/poster/35169`

相关点：
- vision-only autonomous driving。
- 基于 VLM。
- 强调 camera-only，排除昂贵 LiDAR。
- 做 language-action alignment，可连接驾驶理解、视频问答、行为解释。

work3 应借鉴：
- 推理阶段 camera-only 不是缺陷，而是 deployment advantage。
- work3 的输出不应只停留在 retrieval descriptor，而应产生可接 VLM 的 visual/video tokens。
- 可以把 work3 定位为 SimLingo/DriveVLM 前端的几何感知视觉 tokenizer。

### 3.3 如果保留 VPR 主线，可补充第三篇

**Multi-Modal Aerial-Ground Cross-View Place Recognition with Neural ODEs**  
CVPR 2025  
链接：`https://cvpr.thecvf.com/virtual/2025/poster/32913`

相关点：
- multi-modal place recognition。
- camera + LiDAR。
- manifold / neural ODE / multi-domain alignment。
- 使用 KITTI-360 和 nuScenes 构造地点识别 benchmark。

work3 应借鉴：
- 多流形/几何对齐可以继续保留。
- KITTI-360 + nuScenes 的数据选择是合理的。
- 但这篇更偏 VPR，不足以支撑 VLM/video story，优先级低于前两篇。

---

## 4. 方法总体架构

建议把 work3 从单帧 VPR 模型改成三层结构：

```text
Input: B × T × N × 3 × H × W multi-view driving video
       B × T × 1 × H_l × W_l LiDAR range / BEV teacher signal

Camera Student:
  Frame Encoder: DINOv2 ViT-S / ViT-B
  Multi-view Fusion: panoramic token fusion / view-aware attention
  Temporal Encoder: TimeSformer / VideoMAE-style transformer / temporal attention
  Heads:
    1. VPR descriptor head
    2. spatial token head
    3. temporal token head
    4. VLM projector
    5. domain / sensor-failure robustness head

LiDAR Teacher:
  Existing T_net baseline initially
  Later optional: RangeViT / SphereFormer / BEV encoder

Loss:
  L = L_vpr
    + λ_g L_global_geo_distill
    + λ_s L_spatial_token_distill
    + λ_t L_temporal_distill
    + λ_d L_domain / sensor_failure
    + λ_l L_vlm_alignment_optional
```

---

## 5. 必须做的代码改造

### Phase V1 — 输入从单帧变成 clip

| ID | Task | 产出 | Status | Notes |
|---|---|---|---|---|
| V1.1 | 新建 `dataset/NuScenesClipDataset.py`，按 scene/sample token 读取连续 T 帧 | `dataset/NuScenesClipDataset.py` | TODO | 输入输出从 sample-level 变 clip-level |
| V1.2 | 支持 `B × T × N × 3 × H × W` camera 输入 | dataset + model | TODO | N=6，T 默认 4 或 8 |
| V1.3 | 支持 `B × T × 1 × H_l × W_l` LiDAR range 输入 | dataset + teacher | TODO | 先复用原 range image |
| V1.4 | 加 temporal positive/negative mining | dataset / runner | TODO | 原 triplet 只按单帧位置，clip 后可用中心帧 pose |

### Phase V2 — Student 改成 video-token model

| ID | Task | 产出 | Status | Notes |
|---|---|---|---|---|
| V2.1 | 封装 DINOv2 frame encoder | `modules/dinov2_backbone.py` | TODO | 与 work3plan1 P1.1 对齐 |
| V2.2 | 新建 multi-view token fusion 模块 | `modules/multiview_fusion.py` | TODO | 支持 view embedding / camera id embedding |
| V2.3 | 新建 temporal encoder | `modules/temporal_encoder.py` | TODO | 轻量 temporal transformer 即可 |
| V2.4 | 新建 video student | `modules/S_net_video.py` | TODO | 输出 descriptor + spatial_tokens + temporal_tokens |
| V2.5 | 保留原 S_net 作为 baseline | 原文件不动 | TODO | 不能破坏 work3 baseline |

### Phase V3 — 输出变成 VLM-friendly tokens

| ID | Task | 产出 | Status | Notes |
|---|---|---|---|---|
| V3.1 | `forward()` 返回 dict，而不是单纯 tuple | model / runner | TODO | 保留兼容字段 `descriptor`, `domain_logits` |
| V3.2 | 新增 `vlm_projector` | `modules/vlm_projector.py` | TODO | 例如 384/768 → 1024/2048/4096 |
| V3.3 | 输出 `spatial_tokens` | `S_net_video.py` | TODO | 给后续 grounding / VQA |
| V3.4 | 输出 `temporal_tokens` | `S_net_video.py` | TODO | 给后续视频理解 |
| V3.5 | 写 token 保存脚本 | `tools/export_video_tokens.py` | TODO | 后续 VLM 训练可直接加载 |

### Phase V4 — 蒸馏 loss 升级

| ID | Task | 产出 | Status | Notes |
|---|---|---|---|---|
| V4.1 | 保留原多流形 global descriptor distillation | `tools/loss.py` | TODO | 继续作为 global loss |
| V4.2 | 新增 spatial token geometry distillation | `tools/video_distill_loss.py` | TODO | camera patch token 对齐 LiDAR BEV/range token |
| V4.3 | 新增 temporal consistency distillation | `tools/video_distill_loss.py` | TODO | 相邻帧 feature trajectory 一致 |
| V4.4 | 新增 sensor dropout loss | `modules/sensor_dropout.py` + runner | TODO | 对齐 work3plan1 Phase 3 |
| V4.5 | 新增 uncertainty/adaptive weighting | loss 模块 | TODO | 根据 failure / uncertainty 动态调权 |

### Phase V5 — 后续 VLM / 视频理解接口

| ID | Task | 产出 | Status | Notes |
|---|---|---|---|---|
| V5.1 | 构建 driving video caption / QA 数据格式 | `dataset/vlm_export.py` | TODO | 可从 nuScenes metadata + GPT 伪标注开始 |
| V5.2 | 支持导出 Qwen-VL / InternVL 风格 tokens | `tools/export_vlm_features.py` | TODO | 初期只导出 tensor + json |
| V5.3 | 定义 downstream task: video retrieval / event QA / failure reasoning | 文档 + eval | TODO | 先选一个轻量任务 |
| V5.4 | 接一个轻量 VLM baseline | 后续单独计划 | TODO | 不建议第一阶段就端到端训练大模型 |

---

## 6. 实验设计

### 6.1 第一层：保留 VPR

目的：证明升级后没有丢掉 work3 原任务。

实验：
- nuScenes BS → BS
- nuScenes BS → SON
- KITTI-360 train/test split
- Recall@1 / Recall@5 / max F1

### 6.2 第二层：sensor-failure robustness

目的：对齐 CVPR 2025 sensor failure 热点。

协议：
- Full camera views
- Drop 1 camera
- Drop 2 cameras
- Drop 3 cameras
- Front-only camera
- Random occlusion
- Low-light / night subset
- Rain subset
- LiDAR unavailable at inference（默认）

指标：
- Recall@1 under corruption
- Robustness ratio: `R = performance_corrupt / performance_clean`
- Average Robust Recall (ARR)

### 6.3 第三层：视频理解 / VLM 前置任务

先不要直接做大模型训练，建议做轻量验证：

1. **Temporal place/event retrieval**
   - query 是一个短视频 clip。
   - database 是历史 clip。
   - 判断是否同一地点/相似交通场景。

2. **Scene-change understanding**
   - 输入两个 clip，判断场景是否发生显著变化。
   - 可用 ego pose + object annotation 构造弱标签。

3. **Failure-aware retrieval**
   - query 缺失部分相机。
   - database 完整。
   - 测试几何蒸馏 token 是否鲁棒。

4. **Driving QA prototype**
   - 先导出 video tokens。
   - 用小规模自动生成 QA，如：
     - “当前场景是白天还是夜晚？”
     - “是否处于路口？”
     - “前方是否存在车辆/行人？”
     - “相机缺失时模型是否仍能识别相同地点？”

---

## 7. 论文结构建议

### Title

GeoDrive: LiDAR-Privileged Geometry-Aware Visual Pretraining for Robust Driving Video Understanding

### Abstract 主线

1. 自动驾驶 VLM / 视频理解需要稳定的视觉时空表征。
2. 仅用 camera 容易受光照、遮挡、视角缺失影响；LiDAR 有几何优势但部署成本高、可能失效。
3. 提出 LiDAR-privileged training：训练期用 LiDAR 提供几何监督，推理期只用 camera。
4. 方法包含 DINOv2 video student、multi-view temporal fusion、geometry-aware token distillation、sensor dropout。
5. 在 nuScenes / KITTI-360 上验证 VPR、鲁棒性、视频理解前置任务。

### Method 章节

- Problem: LiDAR-privileged camera-only driving video representation learning
- Multi-view Video Student
- LiDAR Geometric Teacher
- Geometry-aware Spatial-Temporal Distillation
- Sensor-Failure-Robust Training
- VLM-oriented Token Projection

### Experiments 章节

- VPR benchmark
- Sensor failure robustness
- Video retrieval / event understanding
- Ablation
  - ResNet vs DINOv2
  - single-frame vs video
  - no distill vs global distill vs token distill vs temporal distill
  - no sensor dropout vs sensor dropout
  - without VLM projector vs with projector

---

## 8. 推荐优先级

### 最推荐路线

先做一个“中等规模但故事完整”的版本：

1. DINOv2 student
2. clip-level dataset
3. temporal encoder
4. descriptor + tokens 双输出
5. LiDAR global + token distillation
6. sensor dropout robustness
7. export VLM tokens

暂时不要一开始就训练完整 VLM。

原因：
- 端到端 VLM 训练算力大、数据构造复杂、不可控。
- 先把 work3 做成 geometry-aware video tokenizer，更稳。
- 后续可以单独开 work4：接 VLM / video QA / driving reasoning。

### 不建议路线

- 只把 ResNet-18 换 DINOv2，然后仍然做单帧 VPR。
  - 这样贡献太小，像工程升级。
- 直接接大 VLM 端到端训练。
  - 风险太高，容易训练不动，也难以证明 LiDAR 蒸馏贡献。
- 同时做 nuScenes + KITTI-360 + Boreas + VLM + world model。
  - 范围失控。

---

## 9. 当前建议给用户的下一步

建议下一轮实际执行以下任务之一：

### Option A：先落地代码骨架

- 新建 `dataset/NuScenesClipDataset.py`
- 新建 `modules/dinov2_backbone.py`
- 新建 `modules/temporal_encoder.py`
- 新建 `modules/S_net_video.py`
- 保证 CPU shape check 可跑

### Option B：先写论文方案

- 新建 `work3_paper_vlm_v1.md`
- 写 abstract / introduction / method overview
- 把 story 从 VPR 改成 VLM/video pretraining

### Option C：先做 related work 表

- 新建 `notes/work3_vlm_related_work.md`
- 整理 CVPR 2025 MoME、CVPR 2025 SimLingo、CVPR 2025 AGPlace、ICCV 2025 SemVPR、ICCV 2025 LiMA 等

我建议优先 Option A，因为代码结构决定后续论文能不能成立。

---

## 10. 会话日志

### 2026-05-17 — Session #2 (VLM/video upgrade planning)
- 用户明确希望把 work3 升级为 VLM / 视频理解等 2026 热点方向。
- 本会话将 work3 新定位为 LiDAR-privileged geometry-aware visual/video pretraining。
- 写入关键对标论文：CVPR 2025 MoME 与 CVPR 2025 SimLingo。
- 给出五阶段代码改造路线：clip dataset、video student、VLM tokens、video distillation、downstream VLM interface。
- 下一步建议：从 Option A 开始落地代码骨架，先做 DINOv2 + clip-level + temporal encoder 的 shape-pass 版本。
