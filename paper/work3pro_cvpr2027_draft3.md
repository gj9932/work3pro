# GeoToken: LiDAR-Privileged Relation Pretraining for Language-Accessible Camera Tokens in Autonomous Driving

Anonymous CVPR Submission

Paper ID: TBD

---

## Abstract

Camera-only driving vision-language models (VLMs) can recognize traffic participants and describe driving scenes, but they remain unreliable in metric and relational spatial reasoning. A model may know that a vehicle is ahead, while failing to estimate whether it is 10 or 30 meters away, whether it occludes another vehicle, whether the left-front area is free, or whether the same object is visible across adjacent cameras. We study whether LiDAR, used only during training, can provide the missing geometric supervision for camera-only driving VLMs.

We propose **GeoToken**, a LiDAR-privileged relation pretraining framework for language-accessible camera tokens. GeoToken does not require LiDAR at inference and does not train a new large VLM. Instead, it pretrains a multi-view camera tokenizer with LiDAR-derived ego-centric spatial relations, then connects the resulting tokens to a frozen driving VLM through a lightweight adapter. The core objective is **Privileged Spatial Relation Distillation (PSRD)**, which converts synchronized LiDAR, ego pose, and calibration into metric, directional, occupancy-topology, and occlusion-order relations, and distills them into camera token interactions. To make the learned geometry accessible to language, GeoToken further introduces **Language-Grounded Spatial Slots (LGSS)**, which align spatial slots with weak semantic-spatial phrases while controlling for template memorization through hard paraphrase splits.

We evaluate GeoToken with driving spatial QA, frozen spatial probing, robustness under sensor degradation, and localization retention. The intended empirical claim is that LiDAR-privileged relation pretraining improves camera-only VLM reasoning about distance, direction, free space, occlusion, cross-view geometry, and temporal motion, while preserving camera-only deployment.

---

## 1. Introduction

Driving VLMs are increasingly used as interfaces for autonomous driving. They can describe traffic scenes, answer driving questions, connect perception to language, and support planning-relevant reasoning. However, a key limitation remains: semantic recognition does not imply reliable spatial understanding. In camera-only driving, the model may correctly identify a car, pedestrian, or intersection, but still fail to reason about distance, bearing, free space, occlusion, cross-view consistency, or whether an object is approaching the ego vehicle.

This gap matters because driving decisions are geometric. It is not enough to know that there is a vehicle in the image. A useful driving representation should also support questions such as: which vehicle is closer to ego? Is the pedestrian on the front-left side or the right side? Is there free space to pass? Is the distant vehicle occluded by the leading car? Is the object visible in both the front and front-left cameras? These are relational questions, not just semantic classification questions.

LiDAR provides accurate 3D structure, but requiring LiDAR at deployment increases cost and system complexity. Many practical data collection pipelines can use LiDAR during training even if the deployed system is camera-only. This creates a privileged learning setting: use LiDAR as a training-time geometric teacher, then deploy only cameras. Existing LiDAR-to-camera distillation methods often match depth maps, BEV features, detector logits, or global descriptors. These targets are useful, but they do not directly match the spatial failures of VLMs, which often concern relations between objects, regions, views, and time steps.

GeoToken addresses this mismatch by using LiDAR to supervise camera token relations rather than asking image tokens to imitate LiDAR features. Given synchronized multi-view images, LiDAR, ego pose, and calibration, we construct an ego-centric spatial relation graph over fixed BEV anchor tokens. Image patches and object boxes provide projected evidence and pseudo labels, but they are not mixed with BEV anchors as graph nodes. The graph contains metric distance, relative direction, occupancy topology, and occlusion order. The camera tokenizer is trained so its token interactions reproduce these relations. Thus, the learned tokens are not literal ego-coordinate points. More precisely, they are camera tokens that encode ego-centric metric and topological cues induced by LiDAR supervision.

Geometry alone is not sufficient for a VLM. A frozen language model also needs an interface through which it can query the spatial representation. We therefore add Language-Grounded Spatial Slots (LGSS), a lightweight slot reader that extracts spatial slots from the pretrained tokens and aligns them with weak spatial phrases such as "a nearby vehicle in front", "free space on the front-left side", and "a pedestrian occluded by the leading car". To reduce the risk that this becomes template memorization, we evaluate under hard paraphrase splits where training phrases and test questions use different linguistic forms.

Our contributions are:

1. We formulate **LiDAR-privileged relation pretraining for language-accessible camera tokens**, targeting camera-only driving VLM spatial reasoning rather than LiDAR deployment or full VLM fine-tuning.
2. We introduce **Privileged Spatial Relation Distillation**, which converts LiDAR observations into metric, directional, occupancy-topology, and occlusion-order relations and distills them into multi-view camera token interactions.
3. We introduce a VLM-centered evaluation protocol with LGSS, hard paraphrase testing, frozen spatial probing, robustness under sensor degradation, and localization retention.

---

## 2. Related Work

### Driving Vision-Language Models

Driving VLMs extend general-purpose VLMs to autonomous driving scene understanding, question answering, and planning-relevant reasoning. DriveLM formulates driving as graph visual question answering, while recent vision-only driving systems show that camera-based VLMs can support closed-loop or action-conditioned behavior. These works demonstrate the promise of VLMs in driving, but they also expose a gap between visual semantics and metric spatial reliability. GeoToken is complementary: it does not replace the VLM, but provides spatially pretrained camera tokens that a frozen driving VLM can access.

### Spatially Grounded VLMs

Spatial VLMs improve 3D, depth, and metric reasoning through spatial supervision, synthetic spatial QA, reconstruction, or explicit coordinate inputs. These methods show that additional spatial supervision can improve VLM reasoning. GeoToken shares this motivation but differs in supervision and deployment: it uses real driving LiDAR only during pretraining and keeps inference camera-only.

### LiDAR-to-Camera Distillation

Cross-modal distillation from LiDAR to camera is widely used for camera-only 3D perception. Prior methods distill depth, BEV features, point features, detector logits, or global descriptors. GeoToken changes the target of distillation. Instead of copying LiDAR features, it distills LiDAR-derived spatial relations into camera token interactions. This target is designed for VLM-style questions such as relative distance, direction, occlusion, and free-space reasoning.

### Robust Camera-Only Driving

Camera-only driving systems are vulnerable to sensor corruption, camera dropout, limited field of view, night, rain, blur, and occlusion. Robustness is therefore not a side detail, but part of the core problem. GeoToken evaluates whether LiDAR-privileged spatial pretraining improves not only clean spatial QA, but also degraded camera-only inference.

---

## 3. Method

### 3.1 Problem Setup

During training, each sample contains a multi-view camera clip, a synchronized LiDAR sequence, ego poses, and camera calibration:

```text
X_cam   = {I_t,n},  t = 1..T, n = 1..N
X_lidar = {P_t},    t = 1..T
```

The goal is to learn a camera-only tokenizer:

```text
Z = F_theta(X_cam),  Z in R^{L x C}
```

where `Z` contains spatially pretrained camera tokens. At inference time, LiDAR is unavailable. The tokenizer receives only camera frames. The tokens are evaluated through frozen VLM spatial QA, frozen spatial probing, robustness tests, and visual place recognition retention.

The intended representation is not a set of explicit 3D coordinates. GeoToken instead uses LiDAR-derived ego-centric relations to make camera tokens encode metric, directional, topological, and occlusion cues.

### 3.2 Camera Spatial Tokenizer

The camera student contains four components:

1. **Frame encoder.** Each camera image is encoded with a pretrained visual backbone such as DINOv2, SigLIP, or CLIP-ViT.
2. **View-aware fusion.** Patch tokens are augmented with camera identity, view geometry, and time embeddings, then fused across cameras.
3. **Temporal encoder.** Adjacent frames are aggregated with a lightweight temporal transformer.
4. **Output heads.** The model outputs BEV anchor tokens, spatial slots, adapter tokens, and a global descriptor for localization retention.

We use four token terms consistently. **Image patch tokens** come from the image backbone. **BEV anchor tokens** are the PSRD-supervised ego-centric tokens. **Spatial slots** are LGSS outputs over BEV anchor tokens. **Adapter tokens** are the projected inputs to the frozen VLM.

For camera image `I_t,n`, the frame encoder produces patch tokens:

```text
H_t,n = E_img(I_t,n)
```

All view tokens are concatenated and passed through view and temporal fusion:

```text
M = E_temp(E_view({H_t,n}))
```

BEV anchor queries then read from `M` through calibration-aware cross-attention, producing BEV anchor tokens `Z`.

A global descriptor is obtained by attention pooling:

```text
g = Pool(Z)
```

### 3.3 LiDAR Ego-Centric Spatial Relation Graph

For each center frame, LiDAR points are transformed into the ego coordinate system using ego pose and calibration. We then rasterize LiDAR-derived geometry onto a fixed ego-centric BEV anchor set. The graph is:

```text
G_lidar = (V, E)
```

To avoid mixing heterogeneous node types, the main PSRD graph uses a single node type: **ego-centric BEV anchor tokens**. We define a fixed local driving range, e.g. `x in [-40m, 40m]` and `y in [-40m, 40m]` in ego coordinates, and partition it into coarse cells such as `2m x 2m`. This produces `40 x 40 = 1600` candidate anchors. Each candidate BEV cell has one anchor query. The multi-view camera encoder produces image features, and each BEV anchor token attends to visible image patches through calibration-aware cross-attention. Object boxes and image patches are not graph nodes in the default PSRD graph; they are used to assign labels, visibility, language phrases, and evaluation regions.

For efficiency, training uses an **active anchor subset**. From the 1600 candidate anchors, we keep at most `L=400` active anchors per frame using a deterministic priority rule: anchors covered by annotated 3D boxes, anchors with occupied LiDAR evidence, anchors on drivable or free-space boundaries, anchors visible in camera frustums, and uniformly sampled remaining free-space anchors. If fewer than 400 anchors satisfy these criteria, we pad with visible free-space anchors. During inference, LiDAR-dependent filtering is not used. We either run the full 1600-anchor grid with chunked attention, or select 400 anchors using camera-only priors predicted by the tokenizer, such as camera-frustum visibility, learned occupancy scores, and learned objectness scores. All reported default training results use the 400-active-anchor setting.

This design gives all graph nodes the same semantics: each node is a fixed ego-centric spatial anchor. Object-level features are obtained by pooling the anchors covered by a 3D box, and free-space regions are obtained by grouping anchors with the same occupancy or drivable label.

Each edge stores privileged spatial labels:

```text
e_ij = (d_ij, a_ij, o_ij, q_ij, m_ij, m^occ_ij)
```

where:

- `d_ij` is pairwise metric distance between BEV anchor centers, discretized into bins.
- `a_ij` is the relative bearing from anchor `i` to anchor `j`.
- `o_ij` is the occupancy-topology relation between anchors.
- `q_ij` is visibility or occlusion order along the camera ray or ego ray.
- `m_ij` is a reliability mask for valid and sufficiently observed relation pairs.
- `m^occ_ij` is a stricter mask for pairs with reliable occlusion-order labels.

The relation graph is the privileged teacher. It is used only during training.

**Relation labels.** We use the following default discretization:

- Distance bins: `[0, 2)`, `[2, 5)`, `[5, 10)`, `[10, 20)`, `[20, 40)`, and `>=40m`.
- Direction bins: 8 ego-plane sectors, `front`, `front-left`, `left`, `rear-left`, `rear`, `rear-right`, `right`, and `front-right`, plus `same-cell` for near-zero displacement.
- Topology classes: `same-free-space`, `same-occupied-component`, `free-to-occupied`, `occupied-to-free`, `different-free-components`, and `unknown`.
- Occlusion classes: `i-before-j`, `j-before-i`, and `same-depth`, computed only when the two anchors project to the same or neighboring camera rays. Pairs without reliable ray overlap have `m^occ_ij = 0` and do not contribute to the occlusion loss.

**Reliability mask.** A pair is valid when both anchors are inside the local driving range, have enough LiDAR support or reliable ray evidence, and are visible in at least one camera after calibration projection. We set `m_ij = 0` for anchors outside the camera field of view, anchors with too few LiDAR points, pairs whose projected rays are ambiguous across cameras, and dynamic objects with inconsistent box association across adjacent frames. For occlusion, we additionally require both anchors to project to the same or adjacent camera-ray bins with separable depth; otherwise `m^occ_ij = 0`.

### 3.4 Privileged Spatial Relation Distillation

PSRD distills LiDAR-derived relations into camera token interactions. Let `Z = {z_i}` denote the BEV anchor tokens produced by calibration-aware cross-attention over multi-view camera features. We compute a camera relation distribution:

```text
R_cam_ij = softmax_j((W_q z_i)^T (W_k z_j) / sqrt(C))
```

The LiDAR relation distribution is constructed from distance and topology:

```text
R_lidar_ij =
  exp(-alpha d_ij) * 1[m_ij = 1] * rho(o_ij)
  ------------------------------------------------
  sum_k exp(-alpha d_ik) * 1[m_ik = 1] * rho(o_ik)
```

where `rho(o_ij)` increases affinity for anchors that belong to the same coherent spatial component. We use:

```text
rho(o_ij) =
  1.5  if same-free-space
  1.3  if same-occupied-component
  0.8  if free-to-occupied or occupied-to-free
  0.5  if different-free-components
  0.0  if unknown
```

`alpha` is set so that the affinity at 20m is approximately `exp(-1)`, i.e. `alpha = 1/20`, and is tuned only on validation data. The relation distribution loss is:

```text
L_rel = sum_i KL(R_lidar_i,N(i) || R_cam_i,N(i))
```

where `N(i)` is the sampled valid edge neighborhood for anchor `i`.

To avoid an `O(L^2)` objective over all BEV anchors, we train on a sparse edge set. For each anchor `i`, we sample at most `P` edges:

- `P_near` nearest valid anchors by metric distance;
- `P_topo` anchors from the same free-space or occupied component;
- `P_occ` valid occlusion-order pairs sharing a camera ray neighborhood;
- `P_hard` hard negatives from different components but similar image appearance;
- `P_rand` random valid long-range anchors.

In the default setting, `L=400` active BEV anchors and `P=32` sampled edges per anchor give at most `12.8k` pairwise edges per frame, rather than `160k` dense pairs over the active set or 2.56M pairs over all 1600 candidates. Evaluation probes may use denser pairs, but training uses the sparse sampled graph.

We further predict pairwise metric, direction, topology, and occlusion relations from sampled token pairs:

```text
d_hat_ij, a_hat_ij, o_hat_ij, q_hat_ij
  = phi([z_i, z_j, z_i - z_j, z_i * z_j])
```

The pairwise losses are:

```text
L_dist = sum_ij m_ij SmoothL1(d_hat_ij, d_ij)
L_dir  = sum_ij m_ij CE(a_hat_ij, a_ij)
L_topo = sum_ij m_ij CE(o_hat_ij, o_ij)
L_occ  = sum_ij m^occ_ij CE(q_hat_ij, q_ij)
```

The PSRD objective is:

```text
L_psrd =
  lambda_rel  L_rel
+ lambda_dist L_dist
+ lambda_dir  L_dir
+ lambda_topo L_topo
+ lambda_occ  L_occ
```

This occlusion term is important. If occlusion is evaluated, the training objective must explicitly supervise visibility order or the method would have a training-evaluation mismatch.

Unless otherwise tuned, we use `lambda_rel=1.0`, `lambda_dist=1.0`, `lambda_dir=0.5`, `lambda_topo=0.5`, and `lambda_occ=0.5`. If class imbalance is severe, `L_topo` and `L_occ` use inverse-frequency class weights computed on the training split.

### 3.5 Language-Grounded Spatial Slots

PSRD makes camera tokens spatially structured, but a frozen VLM still needs an interface to access that structure through language. LGSS introduces `K` learnable slot queries:

```text
S = CrossAttn(Q_slot, Z)
```

LGSS uses three slot groups with separate query embeddings:

- **Object slots** read anchors covered by 3D boxes and dynamic objects.
- **Region slots** read free-space, occupied, drivable, and junction-like areas.
- **Relation slots** read pairs or small groups of object/region slots for closer/farther, left/right, occlusion, and cross-view relations.

This avoids forcing one slot type to represent objects, regions, and relations simultaneously. Object and region slots are matched to pseudo targets before phrase alignment, while relation slots are built from matched object/region slots.

We generate weak spatial phrases from 3D boxes, LiDAR relations, BEV occupancy, and metadata:

- `a nearby vehicle in front`
- `free space on the front-left side`
- `the farther vehicle behind the leading car`
- `a pedestrian occluded by the front vehicle`
- `the same vehicle visible across front and front-left cameras`

LGSS uses a frozen text representation aligned with the target VLM interface. For CLIP-aligned VLMs, we use the paired CLIP text encoder. For LLaVA/InternVL/Qwen-VL-style models, we use the model's own tokenizer and language embedding stack, followed by a lightweight projection into the adapter space. CLIP text embeddings are used only as a controlled ablation when the target VLM is not CLIP-aligned.

Slot-target assignment uses Hungarian matching over a cost that combines spatial overlap, class compatibility, and phrase similarity:

```text
cost(s_k, y_j) =
  lambda_iou  (1 - IoU(A_k, A_j))
+ lambda_cls  CE(c_hat_k, c_j)
+ lambda_txt  (1 - sim(s_k, t_j))
```

where `A_k` is the anchor attention map of slot `s_k`, and `A_j` is the pseudo target mask from 3D boxes or BEV regions. When a phrase can refer to multiple valid objects, such as "a vehicle in front", we use multi-positive contrastive learning:

```text
L_slot = - log  sum_{p in P(i)} exp(sim(s_i, t_p) / tau)
               -----------------------------------------
               sum_j exp(sim(s_i, t_j) / tau)
```

where `P(i)` is the set of positive phrases or targets for slot `i`.

We also train lightweight pairwise QA heads over slots for relation labels such as closer/farther, left/right/front, free/occupied, and occluded/not occluded:

```text
L_qa = CE(h_qa(s_i, s_j, question_type), y_ij)
```

LGSS is not intended to be the source of geometric learning. Its role is to make PSRD-pretrained spatial structure easier for a frozen VLM to access.

To prevent multiple slots from collapsing onto the same object or region, we add a diversity regularizer on slot attention maps:

```text
L_div = || A A^T - I ||_F
```

where each row of `A` is a normalized slot-to-anchor attention map. For videos, matched slots are encouraged to remain temporally consistent after ego-motion compensation:

```text
L_slot_temp = || Warp(A_t, T_t->t+1) - A_t+1 ||_1
```

The LGSS objective is:

```text
L_lgss =
  lambda_slot L_slot
+ lambda_qa L_qa
+ lambda_div L_div
+ lambda_slot_temp L_slot_temp
```

We use `lambda_slot=1.0`, `lambda_qa=1.0`, `lambda_div=0.05`, and `lambda_slot_temp=0.1` by default.

To control template memorization, we use:

- **Template split:** train and test use the same relation categories but disjoint scene instances.
- **Hard paraphrase split:** train phrases and test questions use different linguistic forms.
- **Object-composition split:** train and test use different object-relation combinations.
- **No-LGSS ablation:** evaluates whether gains come from language access or from the BEV anchor tokens themselves.

### 3.6 Auxiliary Objectives

We use auxiliary objectives only to stabilize training and retain useful driving representation properties.

**Localization retention.**

```text
L_vpr = max(0, Delta + s(g, g-) - s(g, g+))
```

**Temporal ego-motion consistency.**

```text
L_temp = CE(r_hat_t,t+k, r_t,t+k)
```

**Sensor degradation consistency.**

We randomly drop cameras, restrict field of view, or apply image corruption. The degraded representation is encouraged to remain consistent with the full-camera representation:

```text
L_cons = || stopgrad(Z_full) - Z_deg ||_2^2
```

The total pretraining objective is:

```text
L_total =
  L_psrd
+ L_lgss
+ lambda_vpr  L_vpr
+ lambda_temp L_temp
+ lambda_cons L_cons
```

### 3.7 Frozen VLM Adapter

To evaluate whether GeoToken is useful for VLMs, we connect pretrained tokens or slots to a frozen driving VLM:

```text
U = A_psi([Z, S])
```

where `A_psi` is an MLP projector or Q-Former-style adapter. During this stage, the camera tokenizer and VLM are frozen. Only the adapter is trained on driving spatial QA. This protocol isolates the value of the pretrained camera tokens and reduces the risk of attributing improvements to full VLM fine-tuning.

---

## 4. Experiments

### 4.1 Datasets

**nuScenes.** We use nuScenes as the primary dataset because it provides six synchronized cameras, LiDAR, ego pose, calibration, weather variation, lighting variation, and driving scene metadata. LiDAR is used only during pretraining. All camera-only evaluations use RGB images only.

**DriveLM-nuScenes and spatial QA.** We evaluate on DriveLM-style annotations where applicable and construct additional spatial QA from nuScenes metadata, 3D boxes, ego pose, and LiDAR-derived relations. Categories include distance comparison, relative direction, free-space reasoning, occlusion, cross-view correspondence, and temporal motion.

**KITTI-360.** We use KITTI-360 for cross-dataset geometry and localization probing. Because KITTI-360 does not match the six-camera nuScenes rig, it is treated as an out-of-distribution evaluation rather than a same-setup benchmark.

### 4.2 Main Evaluation Tasks

#### Driving Spatial QA

A frozen VLM receives either baseline adapter tokens or GeoToken adapter tokens derived from BEV anchor tokens and LGSS slots. Questions are grouped into:

- **Distance:** which object is closer to ego?
- **Direction:** where is the object relative to ego?
- **Topology:** is there free space on the left, right, or front?
- **Occlusion:** is one object blocked by another?
- **Cross-view:** is the same object visible across adjacent cameras?
- **Temporal:** is the object approaching or moving away?

Metrics include overall accuracy, category-wise accuracy, and hard paraphrase accuracy.

#### Frozen Spatial Probing

We freeze the camera tokenizer and train small heads for:

- depth prediction: AbsRel, RMSE, and delta accuracy;
- BEV occupancy or map probing: IoU and mIoU;
- pairwise relation probing: distance-bin, direction-bin, topology, and occlusion-order accuracy;
- localization probing: Recall@1/5/10 and pose error.

#### Robustness

We evaluate clean and corrupted camera-only inputs:

- single camera drop;
- multiple camera drop;
- front-only camera;
- limited field of view;
- random image occlusion;
- night;
- rain;
- motion blur.

We report:

```text
Robustness Ratio = Performance_corrupt / Performance_clean
```

and average robust accuracy across corruptions.

#### VPR Retention

Visual place recognition is used as a retention task. It verifies that spatial token pretraining does not destroy the original localization capability. We report Recall@1/5/10 and max F1.

### 4.3 Baselines

We compare against:

1. Frozen VLM original visual input.
2. CLIP or SigLIP image patch tokens with the same adapter.
3. DINOv2 image patch tokens with the same adapter.
4. DINOv2 plus temporal encoder.
5. DINOv2 plus depth auxiliary pretraining.
6. DINOv2 plus BEV or occupancy auxiliary pretraining.
7. DINOv2 plus sensor dropout.
8. Descriptor-only LiDAR distillation.
9. GeoToken without LGSS.
10. GeoToken with PSRD and LGSS.

For VLM experiments, all methods use the same frozen VLM and the same adapter capacity.

### 4.4 Main Results

**Table 1. Driving spatial QA on nuScenes.** All methods use the same frozen VLM and adapter capacity. Numbers are placeholders.

| Method | Overall | Distance | Direction | Topology | Occlusion | Cross-view | Temporal | Hard Paraphrase |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen VLM original input | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| CLIP adapter | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DINOv2 adapter | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DINOv2 + Depth | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DINOv2 + BEV | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Descriptor-only distillation | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| GeoToken w/o LGSS | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| GeoToken | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

**Table 2. Frozen spatial probing.**

| Method | Depth AbsRel down | BEV IoU up | Relation Acc up | Occlusion Acc up | VPR R@1 up |
|---|---:|---:|---:|---:|---:|
| DINOv2 | TBD | TBD | TBD | TBD | TBD |
| DINOv2 + Depth | TBD | TBD | TBD | TBD | TBD |
| DINOv2 + BEV | TBD | TBD | TBD | TBD | TBD |
| Descriptor-only distillation | TBD | TBD | TBD | TBD | TBD |
| GeoToken | TBD | TBD | TBD | TBD | TBD |

**Table 3. Robustness under camera degradation.**

| Method | Clean QA | Robust QA | QA Ratio | Clean VPR | Robust VPR | VPR Ratio |
|---|---:|---:|---:|---:|---:|---:|
| DINOv2 | TBD | TBD | TBD | TBD | TBD | TBD |
| DINOv2 + dropout | TBD | TBD | TBD | TBD | TBD | TBD |
| Descriptor-only distillation | TBD | TBD | TBD | TBD | TBD | TBD |
| GeoToken | TBD | TBD | TBD | TBD | TBD | TBD |

### 4.5 Ablation Study

We isolate the contribution of each component.

**Table 4. PSRD and LGSS ablation.**

| Variant | Spatial QA | Hard Paraphrase | Relation Probe | Occlusion Probe | BEV IoU | Robust QA |
|---|---:|---:|---:|---:|---:|---:|
| No LiDAR privileged supervision | TBD | TBD | TBD | TBD | TBD | TBD |
| Depth-only distillation | TBD | TBD | TBD | TBD | TBD | TBD |
| BEV-only distillation | TBD | TBD | TBD | TBD | TBD | TBD |
| Descriptor-only distillation | TBD | TBD | TBD | TBD | TBD | TBD |
| PSRD without metric distance | TBD | TBD | TBD | TBD | TBD | TBD |
| PSRD without direction | TBD | TBD | TBD | TBD | TBD | TBD |
| PSRD without topology | TBD | TBD | TBD | TBD | TBD | TBD |
| PSRD without occlusion order | TBD | TBD | TBD | TBD | TBD | TBD |
| PSRD without LGSS | TBD | TBD | TBD | TBD | TBD | TBD |
| LGSS with CLIP text encoder | TBD | TBD | TBD | TBD | TBD | TBD |
| LGSS with mismatched text encoder | TBD | TBD | TBD | TBD | TBD | TBD |
| LGSS with target-VLM text space | TBD | TBD | TBD | TBD | TBD | TBD |
| LGSS with shuffled phrases | TBD | TBD | TBD | TBD | TBD | TBD |
| Full GeoToken | TBD | TBD | TBD | TBD | TBD | TBD |

**Table 5. Architecture and input ablation.**

| Setting | Spatial QA | Robust QA | VPR R@1 |
|---|---:|---:|---:|
| T=1 | TBD | TBD | TBD |
| T=3 | TBD | TBD | TBD |
| T=5 | TBD | TBD | TBD |
| Front-only | TBD | TBD | TBD |
| 3 cameras | TBD | TBD | TBD |
| 6 cameras | TBD | TBD | TBD |
| MLP adapter | TBD | TBD | TBD |
| Q-Former adapter | TBD | TBD | TBD |

### 4.6 Qualitative Analysis

We visualize token affinity maps, slot attention, and LiDAR-derived spatial relation graphs. A successful GeoToken model should assign high affinity to tokens that are close in 3D space even when they are far in the image plane or located in adjacent camera views. We also visualize failures under heavy occlusion, night scenes, missing cameras, and sparse LiDAR supervision.

Suggested figures:

- **Figure 1:** GeoToken training and inference overview: multi-view images → image patch tokens → calibration-aware cross-attention → BEV anchor tokens → PSRD relation graph loss → LGSS slots → frozen VLM adapter.
- **Figure 2:** LiDAR ego-centric relation graph construction.
- **Figure 3:** PSRD losses for relation, distance, direction, topology, and occlusion order.
- **Figure 4:** LGSS and frozen VLM adapter protocol.
- **Figure 5:** Token affinity and slot visualization under clean and degraded inputs.

---

## 5. Discussion

### Why Relations Instead of Dense Depth?

Dense depth supervision improves metric cues, but driving VLM questions are often relational. The model needs to know which object is closer, which region is free, whether one object occludes another, and whether an object is consistent across views. PSRD directly supervises these relations.

### Why Language-Grounded Slots?

Spatial tokens may contain useful geometry, but a frozen VLM may not know how to query them. LGSS provides a lightweight language-access mechanism. The hard paraphrase split is necessary because otherwise the improvement could come from memorizing spatial templates rather than learning language-accessible geometry.

### Why Frozen VLM Evaluation?

Full VLM fine-tuning can hide where gains come from. By freezing both the camera tokenizer and the VLM during adapter training, the evaluation better isolates whether the pretrained tokens are useful.

---

## 6. Limitations

GeoToken requires paired camera-LiDAR data during pretraining. It also depends on calibration quality and LiDAR visibility. Sparse LiDAR, long-range objects, adverse weather, and heavy occlusion may produce incomplete relation labels. LGSS uses weak language phrases and may still carry template bias, so hard paraphrase and object-composition splits are required. Finally, frozen VLM spatial QA is a strong proxy, but it does not replace full closed-loop driving evaluation.

---

## 7. Conclusion

We presented GeoToken, a LiDAR-privileged relation pretraining framework for language-accessible camera tokens in autonomous driving. GeoToken uses LiDAR only during training to supervise ego-centric metric, directional, occupancy-topology, and occlusion-order relations, then deploys using cameras only. Through PSRD and LGSS, the resulting tokens are designed to support frozen driving VLMs on distance, direction, free-space, occlusion, cross-view, and temporal reasoning. The central claim is not that LiDAR is needed at inference, but that training-time LiDAR can teach camera-only models spatial structure that image-language pretraining alone does not reliably provide.

---

## References

[1] nuScenes: A multimodal dataset for autonomous driving. CVPR, 2020.

[2] KITTI-360: A novel dataset and benchmarks for urban scene understanding in 2D and 3D. TPAMI, 2022.

[3] DriveLM: Driving with Graph Visual Question Answering. ECCV, 2024.

[4] SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities. CVPR, 2024.

[5] SimLingo: Vision-Only Closed-Loop Autonomous Driving with Language-Action Alignment. CVPR, 2025.

[6] VisionPAD: A Vision-Centric Pre-training Paradigm for Autonomous Driving. CVPR, 2025.

[7] SpaceDrive: Infusing Spatial Awareness into VLM-based Autonomous Driving. arXiv, 2025.

[8] RoboDriveVLM: A Benchmark and Baseline towards Robust Vision-Language Models for Autonomous Driving. arXiv, 2025.

[9] VLM-3R: Vision-Language Models Augmented with Instruction-Aligned 3D Reconstruction. CVPR, 2026.

[10] DRIVESPATIAL: A Benchmark for Spatiotemporal Intelligence in VLMs for Autonomous Driving. arXiv, 2026.
