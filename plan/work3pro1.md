# work3 Pro 升级计划

> 基线方案见 [`work3pro.md`](work3pro.md)。本文档只记录 **Pro → Pro2** 的增量升级，不重复基线动机与 Phase 1–5 实现细节。
>
> 每个新会话若动 Pro2 相关代码或实验，先读本文件；有结论后更新文末 `## 会话日志`。

---

## 0. 版本谱系

| 版本 | 核心增量 | 与 VLM 的关系 |
|---|---|---|
| **work3 Pro** | LiDAR 特权几何蒸馏 → geometry-aware camera tokens | token 有几何/深度/BEV 感，但语义-空间绑定弱，LLM 难直接“读懂” |
| **work3 Pro2-B** | 在几何蒸馏上叠加 **弱语言监督** | token 同时携带 **可被 LLM 索引的语义-空间绑定**，更贴 VLM 消费方式 |
| **work3 Pro2-Space**（对标项） | 与 SpaceDrive 对齐的 3D PE / 接口与评测协议 | 验证“蒸馏来的空间感”能否达到或补充 explicit 3D PE 路线 |

关系一句话：

> Pro 教 camera token **几何在哪**；Pro2-B 再教 **什么在哪、用什么语言描述**；Pro2-Space 用 SpaceDrive 的 PE+规划范式做 **可发表的对标与组合实验**。

---

## 1. work3 Pro2-B — 弱语言监督的几何蒸馏

### 1.1 要解决的问题

work3 Pro 的 `vlm_tokens` / `spatial_tokens` 主要通过 LiDAR 几何损失塑形，具备：

- 深度排序、BEV 布局、token 间几何关系、时序 ego-motion 等 **隐式坐标感**；

但 VLM 消费视觉信息时，更依赖 **语义锚点 + 空间指称** 的联合表征，例如：

- “前方 15m 的轿车” vs “右侧路沿 3m”；
- “更近的是左侧行人还是前方车辆”。

仅靠几何蒸馏，token 对 LLM 仍像 **连续几何场**，缺少 **离散语义-空间槽位（semantic-spatial slot）**，与 SpaceDrive 强调的 “index specific visual semantics in spatial reasoning” 仍有 gap。

**Pro2-B 目标**：训练阶段用 **低成本弱文本**，把几何 token 与 **可读的空间-语义描述** 对齐，推理仍 camera-only，不依赖 LiDAR 或人工标注。

### 1.2 弱语言监督来源（训练期 only）

优先级从高到低：

1. **nuScenes 结构化元数据模板化**
   - 3D box 类别、距离分桶、方位词（前/后/左/右）、车道/路口 scene tag。
   - 模板示例：`{方位} {距离桶} 处有 {类别}`、`自车前方 {N} 个可检测目标`。

2. **投影对齐的 patch–phrase 弱对齐**
   - LiDAR/depth 投影到各 view patch；每个 patch 绑定 top-1 类别 + 距离区间文本。
   - 不追求逐像素 caption，只做 **token 级 phrase assignment**。

3. **自动空间关系 QA（伪标签）**
   - 从 box 深度/横向坐标自动生成比较问句与答案：
     - Q: 前方车辆与右侧车辆哪个更近？ A: 前方。
   - 用于 contrastive / ITC 式对齐，而非端到端生成训练。

4. **（可选，后期）驾驶场景 caption 模型伪标注**
   - 仅作辅助，权重低于 1–3，避免 caption 噪声盖过几何 teacher。

**原则**：全部是 **弱、模板化、可自动批量构造**；不做昂贵人工 VQA 标注。

### 1.3 方法骨架

在 Pro 的 camera student 上增加 **Language Alignment Head**，与现有几何头并行：

```text
Multi-view clip
  └─► Camera Student (DINOv2 + fusion + temporal) ──► spatial_tokens / vlm_tokens
         │                              │
         │                              ├──► Geo heads (depth / BEV / relation)  ← 继承 Pro
         │                              └──► Lang-Geo head ──► slot_tokens [B, K, C]
         │
LiDAR Teacher ──► 几何蒸馏损失 (Pro)
         │
Weak text {phrase_k, qa_pair} ──► Text Encoder (frozen MiniLM / CLIP-T / 小 LLM embedding)
         │
         └──► 对齐损失：slot_tokens ↔ phrase / answer
```

**slot_tokens**：固定 K 个可学习 query（或从 spatial_tokens 聚类读出），每个 slot 对应一个 **语义-空间槽**，便于后续 VLM cross-attention。

### 1.4 损失设计（在 Pro `L_total` 上增量）

```text
L_Pro2B = L_total_Pro
        + λ_lc L_lang_contrast      # slot ↔ phrase ITC / InfoNCE
        + λ_lr L_lang_relation       # 空间比较 QA 的分类或匹配损失
        + λ_lg L_lang_geo_consist    # 同一 slot 的几何预测与文本距离桶一致
```

| 损失 | 作用 | 实现要点 |
|---|---|---|
| `L_lang_contrast` | phrase 与 slot 绑定 | 正样本：同帧投影匹配的 (slot, phrase)；负样本：同 batch 其他 phrase |
| `L_lang_relation` | 可读的比较推理 | QA 答案 embedding 与 `[slot_i, slot_j]` 组合向量对齐 |
| `L_lang_geo_consist` | 防止语言对齐漂移到纯语义 | 文本中的距离桶需与 LiDAR/depth 蒸馏的深度分桶一致，不一致则降权 |

**推理阶段**：去掉 text encoder；`slot_tokens` 与 `vlm_tokens` 一并导出，供下游 VLM 冻结或轻量 projector 使用。

### 1.5 预期能力变化（相对 Pro）

| 维度 | Pro | Pro2-B |
|---|---|---|
| Depth / BEV probing | ✓ | ✓（保持） |
| Token 语义-空间可读性 | 弱 | 强：frozen VLM 上 **spatial referring** 提升 |
| 接 SpaceDrive / SimLingo | 仅几何 token | 几何 + **语言对齐 slot**，更易接 PE 或 text prompt |
| 训练成本 | 中 | +小 text encoder forward + 伪标签管线 |

### 1.6 代码增量（相对 Pro Phase 2–4）

| 模块 | 路径建议 |
|---|---|
| 弱标签构造 | `tools/build_weak_lang_labels.py` |
| Text encoder（冻结） | `modules/text_encoder_frozen.py` |
| Lang-Geo head | `modules/lang_geo_head.py` |
| 损失 | `tools/lang_geo_loss.py` |
| 训练入口 | `tools/train_pro2b.py`（继承 `train_video.py`） |
| 评测 | `tools/eval_spatial_referring.py`、`tools/eval_lang_geo_qa.py` |

### 1.7 Pro2-B MVP（最小可验证）

1. nuScenes clip + 模板 phrase（仅用 box 元数据）。
2. Pro 已有 global + spatial token distill 跑通。
3. 加 `L_lang_contrast`，K=8 slot_tokens。
4. 评测：
   - **Spatial Referring Acc**：给定 phrase，选正确 patch/slot 的比例；
   - **Weak QA Acc**：自动生成比较题；
   - 原有 depth probing / VPR 不掉点（证明未牺牲几何）。

---

## 2. 对标 SpaceDrive

**论文**：*SpaceDrive: Infusing Spatial Awareness into VLM-based Autonomous Driving*（CVPR 2026，[arXiv:2512.10719](https://arxiv.org/abs/2512.10719)，[代码](https://github.com/zhenghao2519/SpaceDrive)）

### 2.1 SpaceDrive 在做什么

| 要点 | 内容 |
|---|---|
| 问题 | VLM 用逐 digit 文本表示坐标，**细粒度 3D 空间关系** 弱 |
| 核心 | **Universal 3D Positional Encoding (PE)**：深度估计 + 历史 ego → metric 3D 坐标 → PE |
| 注入方式 | 3D PE **叠加**到对应 2D visual token；输入/输出用 PE 替代数字 token |
| 下游 | VLM 联合语义+空间推理；轨迹用 **回归头** 而非逐 token 生成数字 |
| 结果 | nuScenes open-loop SOTA 类；Bench2Drive DS **78.02**（VLM 类第二） |

### 2.2 work3 Pro 与 SpaceDrive 的差异（论文叙事用）

| | **SpaceDrive** | **work3 Pro / Pro2** |
|---|---|---|
| 空间信息来源 | 在线多视图 **深度估计** + ego 状态 | 训练期 **LiDAR 特权 teacher** 蒸馏 |
| 表示形式 | 显式 metric **3D PE** 叠加 | 隐式/显式几何 token +（Pro2-B）语义-空间 slot |
| 训练阶段传感器 | 推理可 camera-only，但 PE 依赖深度网络 | 训练用 LiDAR，推理 **纯 camera** |
| 与 VLM 耦合 | 端到端 driving VLM 框架 | **预训练 tokenizer**，再接任意 VLM |
| 弱项互补 | 深度估计误差会传导到 PE | 不直接输出 metric PE，需接 head 或 PE 层 |

**定位句（可写进论文）**：

> SpaceDrive 在 **VLM 内部** 用 explicit 3D PE 修复空间推理；work3 Pro 在 **VLM 之前** 用 LiDAR-privileged distillation 预注入几何与（Pro2-B）语义-空间绑定。二者正交，可组合。

### 2.3 对标实验矩阵（Pro2-Space）

| 实验 ID | 设置 | 目的 |
|---|---|---|
| **S0** | SpaceDrive 官方 / 公开 checkpoint 复现 | 对齐 nuScenes open-loop 与报告指标 |
| **S1** | 仅 Pro `vlm_tokens` → 接 SpaceDrive 同款 VLM backbone（冻结 PE 模块） | 看 **预训练几何 token** 能否替代部分深度-PE 计算 |
| **S2** | Pro2-B `slot_tokens` + SpaceDrive PE | 语义-空间 slot 是否与 explicit PE **叠加增益** |
| **S3** | Pro / Pro2-B vs SpaceDrive depth：depth probing、spatial QA、referring | 不跑全闭环时仍可比 **空间理解** 子能力 |
| **S4**（资源允许） | Bench2Drive 闭环 | 验证组合是否提升 DS / success rate |

**优先指标**（与 SpaceDrive 论文对齐部分）：

- nuScenes open-loop：L2、碰撞率、轨迹相关指标（按官方评测脚本）。
- 空间子任务：spatial referring、weak spatial QA、depth/BEV probing（Pro 独有、SpaceDrive 未强调的可加分项）。

### 2.4 可从 SpaceDrive 吸收到 work3 的模块

1. **Universal 3D PE 编码器**（实现参考官方 repo）
   - 导出 Pro/Pro2-B 的 `spatial_tokens` 后，**叠加同一 PE**，作为 `vlm_tokens` 的可选分支，而非替换 LiDAR 蒸馏。

2. **“坐标不走 digit token” 的输出协议**
   - Pro2-B 的 slot 输出 + 小回归头，对接轨迹/waypoint，与 SpaceDrive 的 regression head 叙事一致。

3. **评测协议**
   - 直接复用其 nuScenes open-loop 配置；spatial QA 用我们 weak QA 集补充。

4. **不做 / 延后**
   - 第一版不端到端训大 VLM；不把 SpaceDrive 全链路塞进 Pro 代码库。
   - 先 **tokenizer + probing + open-loop 接口对齐**，闭环放 Pro2-Space S4。

### 2.5 组合架构（目标态）

```text
Training:
  Camera clip ──► work3 Pro2-B Student ──► {vlm_tokens, slot_tokens}
  LiDAR ──► Teacher ──► L_geo (Pro)
  Weak text ──► L_lang (Pro2-B)

  [可选分支] metric 3D coords (from teacher, not estimated depth)
         └──► Universal 3D PE ──► 叠加到 vlm_tokens

Inference (camera-only):
  Camera clip ──► Student tokens (+ optional PE from student depth head)
         └──► Frozen / LoRA VLM + regression head (SpaceDrive-style)
```

---

## 3. 升级项 3 — 与 SimLingo 的接口（Pro2 下游）

| 项 | 内容 |
|---|---|
| 动机 | SimLingo 强 **language-action**，弱几何；Pro2-B 的 slot_tokens 可作 **几何前端** |
| 实验 | Pro2-B tokens 替换 SimLingo 视觉编码器输入 → 看 L-A 对齐任务是否提升 |
| 优先级 | 低于 Pro2-B 与 SpaceDrive S1–S3 |

---

## 4. 升级项 4 — Sensor-failure × 语言鲁棒（Pro2-B+）

在 Pro Phase 3 sensor dropout 基础上：

- 训练时对 **phrase 对应的 view** 做 camera drop，加 `L_consistency` 约束 slot 语义不变；
- 评测：**corrupt 下 Spatial Referring / QA** 与 VPR 的 Robustness Ratio（继承 Pro 9.3）。

---

## 5. 推荐实施顺序

```text
Pro Phase 1–2（video student + 几何蒸馏）     ← 必须先于 Pro2-B
    ↓
Pro2-B MVP（模板 phrase + L_lang_contrast）   ← 本文件核心增量 1
    ↓
Pro2-Space S1–S3（SpaceDrive 对标）            ← 本文件核心增量 2
    ↓
Pro Phase 3–4（sensor + vlm export）与 Pro2-B 合并
    ↓
Pro2-Space S4 / SimLingo 下游（资源允许）
```

---

## 6. 论文贡献点（Pro2 增量，相对 Pro 四条）

1. **Weak language-supervised geometric distillation**：用自动 phrase/QA 把 LiDAR 几何 token 绑定到 LLM 可读的语义-空间槽。
2. **Complementary to explicit 3D PE (SpaceDrive)**：预训练 tokenizer + PE 组合优于单独一路。
3. **Spatial referring & weak QA benchmark**：补充 VPR/depth probing，证明“不仅有几何，还能被语言索引”。

---

## 7. 风险

| 风险 | 应对 |
|---|---|
| 弱标签噪声 | 以几何分桶一致性过滤；`λ_lg` 约束 |
| 语言损失冲淡几何 | 先训 Pro 再开 `λ_lc`；或 warmup |
| SpaceDrive 复现成本高 | 先做 S3 子任务对标 + S1 接口；S4 可选 |
| 范围膨胀 | Pro2-B MVP 不含端到端 VLM 训练 |

---

## 8. 会话日志

### 2026-05-24 — 初始化 Pro 升级计划

- 新建本文档，相对 `work3pro.md` 记录 **Pro2-B（弱语言监督几何蒸馏）** 与 **SpaceDrive 对标路线（Pro2-Space）**。
- 明确：Pro 负责几何；Pro2-B 负责语义-空间绑定；SpaceDrive 对标负责 explicit PE 与 open-loop 协议对齐。
- 下一步：Pro Phase 1–2 跑通后，实现 `build_weak_lang_labels.py` + `lang_geo_head.py` 做 Pro2-B MVP。

