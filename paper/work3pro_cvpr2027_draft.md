# GeoToken: LiDAR-Privileged Spatial Token Pretraining for Camera-Only Driving VLMs

Anonymous CVPR Submission

Paper ID: TBD

---

## Abstract

Camera-only driving vision-language models (VLMs) have shown promising semantic understanding and language-action alignment, but they remain unreliable in metric spatial reasoning, cross-view geometry, and adverse sensing conditions. This paper studies whether training-time LiDAR can be used as a privileged spatial teacher to produce camera-only visual tokens that are more useful for driving VLMs. We propose **GeoToken**, a LiDAR-privileged pretraining framework for multi-view camera video representation learning. Instead of directly regressing LiDAR features, GeoToken introduces **Privileged Spatial Relation Distillation (PSRD)**, which converts LiDAR observations into metric, directional, and occupancy-topology relations, and distills these relations into camera token interactions. At inference time, GeoToken uses only multi-view camera frames and outputs spatial tokens that can be connected to a frozen VLM through a lightweight adapter. We evaluate GeoToken on driving spatial question answering, frozen spatial probing, and robustness under sensor degradation. The intended empirical claim is that LiDAR-privileged spatial relation pretraining improves camera-only spatial reasoning, cross-view grounding, and degradation robustness. If validated, this would suggest that training-time 3D sensing provides a practical path toward spatially grounded camera-only driving VLMs without requiring LiDAR at deployment.

---

## 1. Introduction

Vision-language models are becoming an increasingly important interface for autonomous driving. Recent driving VLMs can describe traffic scenes, answer driving questions, align language with actions, and even participate in closed-loop driving. However, a central limitation remains: image-language pretraining does not naturally provide reliable metric spatial understanding. A model may correctly identify a vehicle, pedestrian, or intersection, but still fail to reason about distance, bearing, occlusion, free space, or whether an object is approaching the ego vehicle. These errors are particularly important in driving, where decisions depend not only on semantic recognition but also on geometry.

This limitation has become more visible as driving VLMs move from scene captioning toward language-action alignment and planning-relevant reasoning. Vision-only systems such as SimLingo demonstrate that camera-only VLMs can support closed-loop driving, while spatially grounded driving VLMs such as SpaceDrive highlight the need to explicitly inject spatial awareness into VLM-based autonomous driving. In parallel, robust VLM benchmarks for autonomous driving show that sensor corruption and environmental degradation can substantially affect decision quality. These trends motivate a representation learning question: can a camera-only VLM receive better visual tokens if those tokens are pretrained with privileged 3D geometry?

LiDAR provides accurate depth and spatial structure, but requiring LiDAR at deployment increases cost and system complexity. In many practical settings, however, LiDAR is available during data collection and training. This creates a natural privileged learning setup: use LiDAR as a training-time spatial examiner, then deploy only cameras. Existing LiDAR-to-camera distillation methods often align global descriptors, depth maps, or BEV features. While useful, these objectives do not directly target the relational spatial failures of VLMs, such as comparing distances between objects, reasoning across views, or understanding occupancy topology.

We propose **GeoToken**, a LiDAR-privileged spatial token pretraining framework for camera-only driving VLMs. GeoToken trains a multi-view camera video tokenizer whose outputs can be used for VLM spatial reasoning, probing, and localization. The core of GeoToken is **Privileged Spatial Relation Distillation (PSRD)**. PSRD first converts LiDAR observations and ego pose into a spatial relation graph over visible camera tokens. The graph encodes metric distance, relative direction, visibility, occlusion order, and occupancy-topology relations. The camera tokenizer is then trained to reproduce these privileged relations through token-token interactions. Thus, the student does not simply copy LiDAR features; it learns how visual tokens should be organized according to 3D driving geometry.

At inference time, GeoToken takes only multi-view camera video. The resulting spatial tokens can be fed into a frozen VLM using a lightweight adapter, or evaluated directly through frozen probing tasks. This design keeps the method compatible with existing driving VLMs and avoids the cost of full VLM fine-tuning in the first stage.

Our contributions are:

1. We formulate **LiDAR-privileged spatial token pretraining** for camera-only driving VLMs, targeting spatial reasoning and robustness rather than only perception or place recognition.
2. We introduce **Privileged Spatial Relation Distillation**, which distills LiDAR-derived metric, directional, and occupancy-topology relations into multi-view camera token interactions.
3. We propose a VLM-centered evaluation protocol covering driving spatial QA, frozen depth/BEV/relation probing, robustness under camera degradation and adverse conditions, and visual place recognition retention.

---

## 2. Related Work

### Driving Vision-Language Models

Driving VLMs extend general-purpose vision-language models to autonomous driving scenarios. DriveLM formulates driving as graph-based visual question answering. DriveVLM and related systems use VLMs for scene understanding and planning. SimLingo further shows that a vision-only VLM can support closed-loop driving through language-action alignment. These works demonstrate the promise of language-conditioned driving models, but also expose a gap between semantic understanding and metric spatial reasoning. GeoToken is complementary to these systems: it does not replace the VLM, but provides spatially pretrained camera tokens that can be connected to a frozen driving VLM.

### Spatially Grounded VLMs

Recent work on spatial VLMs aims to improve 3D, depth, and metric reasoning in VLMs. SpatialVLM-style methods use spatial supervision or synthetic spatial QA to improve quantitative reasoning. VLM-3R-style methods augment VLMs with 3D reconstruction signals. SpaceDrive specifically targets spatial awareness in VLM-based autonomous driving. GeoToken follows the same broad motivation but differs in its supervision source and deployment setting: LiDAR is used only during pretraining, while inference remains camera-only.

### Vision-Centric Pretraining for Autonomous Driving

Vision-centric driving pretraining has been studied for 3D detection, occupancy prediction, map segmentation, motion estimation, and BEV representation learning. Methods such as VisionPAD show that geometric pretraining can improve downstream perception. Many approaches reconstruct images, predict future frames, estimate depth, or learn BEV features. GeoToken differs by targeting VLM-usable token representations and by distilling relational 3D structure rather than only dense depth or BEV values.

### Cross-Modal Distillation from LiDAR to Camera

LiDAR-to-camera distillation is a common approach for improving camera-only perception. Prior methods distill global descriptors, point features, depth maps, BEV features, or detection logits. In visual place recognition, LiDAR teachers can improve camera descriptors under challenging illumination and viewpoint changes. GeoToken builds on this idea but changes the distillation target from feature matching to spatial relation matching. This is important for VLMs, where many failures are relational: which object is closer, where it lies relative to the ego vehicle, whether it is occluded, and how free space is structured.

### Robust Driving VLM Evaluation

Robustness is critical for camera-only driving. Recent benchmarks evaluate VLM-based driving systems under sensor corruption, environmental degradation, and prompt perturbation. GeoToken includes robustness as a main evaluation axis. We test whether LiDAR-privileged pretraining improves not only clean spatial QA but also performance under camera drop, limited field of view, occlusion, night, rain, and motion blur.

---

## 3. Method

### 3.1 Problem Setup

During training, each sample contains a multi-view camera clip, a synchronized LiDAR sequence, ego poses, and camera calibration:

$$
X^{cam} = \{I_{t,n}\}_{t=1,n=1}^{T,N}, \quad
X^{lidar} = \{P_t\}_{t=1}^{T},
$$

where \(T\) is the number of frames and \(N\) is the number of cameras. The training objective is to learn a camera-only tokenizer \(F_\theta\) that outputs spatial tokens:

$$
Z = F_\theta(X^{cam}) \in \mathbb{R}^{L \times C}.
$$

At inference time, LiDAR is unavailable. The tokenizer must operate using only \(X^{cam}\). The output tokens are used for four evaluation families: VLM spatial QA, frozen spatial probing, robustness evaluation, and visual place recognition.

### 3.2 Camera Spatial Tokenizer

The camera student contains four components:

1. **Frame encoder.** Each camera image is encoded by a pretrained visual backbone such as DINOv2, SigLIP, or CLIP-ViT.
2. **View-aware fusion.** Patch tokens are augmented with camera-ID and view-position embeddings, then fused across cameras by view-aware attention.
3. **Temporal encoder.** Tokens from adjacent frames are aggregated by a lightweight temporal transformer.
4. **Task heads.** The model outputs spatial tokens, temporal tokens, a global descriptor for localization, and optional BEV tokens for probing.

Formally, for image \(I_{t,n}\), the frame encoder produces patch tokens:

$$
H_{t,n} = E_{img}(I_{t,n}) \in \mathbb{R}^{L_p \times C}.
$$

After adding camera and time embeddings, all view tokens are concatenated and passed to the fusion and temporal modules:

$$
Z = E_{temp}(E_{view}(\{H_{t,n}\}_{t,n})).
$$

The global descriptor is obtained by attention pooling:

$$
g = \mathrm{Pool}(Z) \in \mathbb{R}^{D}.
$$

### 3.3 LiDAR Privileged Spatial Graph

The LiDAR teacher is not required to be a large 3D network. In the minimal version, LiDAR and ego pose are used to build privileged spatial labels. Stronger LiDAR encoders can be added later as an upper bound.

For each center frame, LiDAR points are transformed into the ego coordinate system and projected into camera views. A visible camera token corresponds to either an image patch with projected LiDAR points or a BEV cell visible in one or more cameras. We construct a graph:

$$
G^{lidar} = (V, E),
$$

where each node \(v_i \in V\) corresponds to a camera token or projected BEV cell. Each edge \(e_{ij}\) stores privileged 3D relations:

$$
e_{ij} = (d_{ij}, a_{ij}, o_{ij}, q_{ij}, m_{ij}).
$$

Here:

- \(d_{ij}\) is metric distance or distance bin between nodes.
- \(a_{ij}\) is relative direction or bearing bin.
- \(o_{ij}\) indicates occupancy-topology relation, such as same free-space region or same occupied component.
- \(q_{ij}\) indicates occlusion or visibility order.
- \(m_{ij}\) is a validity mask for projected and reliable pairs.

This graph is the privileged spatial supervision used by PSRD.

### 3.4 Privileged Spatial Relation Distillation

The key idea of PSRD is to distill LiDAR-derived relations into camera token interactions. Given camera tokens \(Z = \{z_i\}_{i=1}^{L}\), the camera relation matrix is:

$$
R^{cam}_{ij} =
\mathrm{softmax}_j
\left(
\frac{(W_q z_i)^\top (W_k z_j)}{\sqrt{C}}
\right).
$$

The LiDAR relation matrix is computed from 3D distance and occupancy topology:

$$
R^{lidar}_{ij} =
\frac{
\exp(-\alpha d_{ij}) \cdot \mathbb{1}[m_{ij}=1] \cdot \rho(o_{ij})
}{
\sum_k \exp(-\alpha d_{ik}) \cdot \mathbb{1}[m_{ik}=1] \cdot \rho(o_{ik})
},
$$

where \(\rho(o_{ij})\) increases affinity for tokens in the same free-space or occupied region. The relation distillation loss is:

$$
\mathcal{L}_{rel}
=
\sum_i
\mathrm{KL}
\left(
R^{lidar}_{i,:}
\| R^{cam}_{i,:}
\right).
$$

We further predict pairwise metric and directional relations from camera token pairs:

$$
\hat{d}_{ij}, \hat{a}_{ij}, \hat{o}_{ij}
=
\phi([z_i, z_j, z_i-z_j, z_i \odot z_j]).
$$

The metric, direction, and topology losses are:

$$
\mathcal{L}_{dist}
=
\sum_{i,j} m_{ij}
\mathrm{SmoothL1}(\hat{d}_{ij}, d_{ij}),
$$

$$
\mathcal{L}_{dir}
=
\sum_{i,j} m_{ij}
\mathrm{CE}(\hat{a}_{ij}, a_{ij}),
$$

$$
\mathcal{L}_{topo}
=
\sum_{i,j} m_{ij}
\mathrm{CE}(\hat{o}_{ij}, o_{ij}).
$$

The PSRD objective is:

$$
\mathcal{L}_{psrd}
=
\lambda_{rel}\mathcal{L}_{rel}
+ \lambda_{dist}\mathcal{L}_{dist}
+ \lambda_{dir}\mathcal{L}_{dir}
+ \lambda_{topo}\mathcal{L}_{topo}.
$$

Unlike depth regression, PSRD supervises token-token relations. Unlike direct feature distillation, PSRD does not require camera tokens to imitate LiDAR features. The student learns the spatial organization induced by LiDAR.

### 3.5 Auxiliary Objectives

We use three auxiliary objectives to stabilize training and retain useful driving representation properties.

**Localization retention.** A global descriptor \(g\) is trained with a place recognition loss:

$$
\mathcal{L}_{vpr} =
\max(0, \Delta + s(g, g^-) - s(g, g^+)),
$$

where \(g^+\) and \(g^-\) are positive and negative place descriptors.

**Temporal ego-motion consistency.** For adjacent frames, the model predicts relative ego-motion bins or contrastive temporal consistency:

$$
\mathcal{L}_{temp}
=
\mathrm{CE}(\hat{r}_{t,t+k}, r_{t,t+k}).
$$

**Sensor degradation consistency.** We randomly drop cameras, restrict field of view, or apply image corruption during training. The degraded-token representation is encouraged to remain consistent with the full-camera representation:

$$
\mathcal{L}_{cons}
=
\| \mathrm{sg}(Z^{full}) - Z^{deg} \|_2^2.
$$

The total pretraining loss is:

$$
\mathcal{L}
=
\mathcal{L}_{psrd}
+ \lambda_{vpr}\mathcal{L}_{vpr}
+ \lambda_{temp}\mathcal{L}_{temp}
+ \lambda_{cons}\mathcal{L}_{cons}.
$$

### 3.6 VLM Adapter

To evaluate whether GeoToken is useful for VLMs, we connect pretrained spatial tokens to a frozen driving VLM through a lightweight adapter:

$$
U = A_\psi(Z),
$$

where \(A_\psi\) is an MLP projector or Q-Former-style adapter. During this stage, the camera tokenizer and VLM are frozen, and only \(A_\psi\) is trained on driving spatial QA. This protocol isolates the value of the spatial tokens and avoids attributing gains to full VLM fine-tuning.

---

## 4. Experiments

### 4.1 Datasets

**nuScenes.** We use nuScenes as the primary dataset because it provides six synchronized cameras, LiDAR, ego pose, weather and lighting variation, and DriveLM-style language annotations. LiDAR is used only during pretraining. Inference and all camera-only evaluations use only RGB images.

**KITTI-360.** We use KITTI-360 for cross-dataset generalization in localization and spatial probing. Since KITTI-360 does not match the six-camera nuScenes setup, we evaluate it as an out-of-distribution geometry and localization benchmark rather than forcing the same view configuration.

**Driving spatial QA.** We evaluate on DriveLM-nuScenes where applicable and construct additional spatial QA from nuScenes metadata and LiDAR-derived relations. The QA categories include distance comparison, relative direction, free-space reasoning, occlusion, cross-view correspondence, and temporal motion.

### 4.2 Evaluation Tasks

#### Driving Spatial QA

This is the main evaluation. A frozen VLM receives either standard visual tokens, baseline adapter tokens, or GeoToken spatial tokens. Questions are grouped into:

- **Distance:** which object is closer to the ego vehicle?
- **Direction:** where is the object relative to ego?
- **Topology:** is there free space on the left/right/front?
- **Occlusion:** is an object blocked by another object?
- **Cross-view:** is the same object visible across adjacent cameras?
- **Temporal:** is an object approaching or moving away?

Metrics include overall accuracy and category-wise accuracy.

#### Frozen Spatial Probing

We freeze the pretrained camera tokenizer and train small heads for:

- depth prediction: AbsRel, RMSE, and \(\delta\) accuracy;
- BEV occupancy or map probing: IoU and mIoU;
- pairwise relation probing: distance-bin, direction-bin, and occlusion-order accuracy;
- localization probing: Recall@1/5/10 and pose error.

#### Robustness

We evaluate clean and corrupted inputs:

- single camera drop;
- multiple camera drop;
- front-only camera;
- limited field of view;
- random occlusion;
- night;
- rain;
- motion blur.

We report:

$$
\mathrm{Robustness\ Ratio}
=
\frac{\mathrm{Performance}_{corrupt}}
{\mathrm{Performance}_{clean}},
$$

and average robust accuracy across corruptions.

#### Visual Place Recognition

VPR is used as a retention task to verify that spatial token pretraining does not destroy the original work3 localization objective. We report Recall@1/5/10 and max F1.

### 4.3 Baselines

We compare against:

1. CLIP / SigLIP visual tokens.
2. DINOv2 visual tokens.
3. DINOv2 + temporal encoder.
4. DINOv2 + depth auxiliary pretraining.
5. DINOv2 + BEV auxiliary pretraining.
6. DINOv2 + sensor dropout.
7. Descriptor-only LiDAR distillation, matching the original work3-style setup.
8. GeoToken with PSRD.

For VLM experiments, all methods use the same frozen VLM and the same adapter capacity.

### 4.4 Main Results

#### Driving Spatial QA

Table 1 reports spatial QA performance. GeoToken is expected to improve most on distance, direction, topology, and occlusion categories, where LiDAR-derived relation supervision directly matches the evaluation target.

**Table 1. Driving spatial QA on nuScenes.** All methods use the same frozen VLM and adapter capacity. Numbers are placeholders for experiments.

| Method | Overall | Distance | Direction | Topology | Occlusion | Cross-view | Temporal |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen VLM original tokens | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| CLIP adapter | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DINOv2 adapter | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DINOv2 + Depth | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DINOv2 + BEV | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Descriptor-only distillation | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| GeoToken | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

#### Frozen Spatial Probing

Table 2 evaluates whether spatial information is present in frozen tokens without VLM fine-tuning.

**Table 2. Frozen spatial probing.**

| Method | Depth AbsRel down | BEV IoU up | Relation Acc up | VPR R@1 up |
|---|---:|---:|---:|---:|
| DINOv2 | TBD | TBD | TBD | TBD |
| DINOv2 + Depth | TBD | TBD | TBD | TBD |
| DINOv2 + BEV | TBD | TBD | TBD | TBD |
| Descriptor-only distillation | TBD | TBD | TBD | TBD |
| GeoToken | TBD | TBD | TBD | TBD |

#### Robustness

Table 3 reports average performance under camera and weather degradation.

**Table 3. Robustness under sensor degradation.**

| Method | Clean QA | Robust QA | QA Ratio | Clean VPR | Robust VPR | VPR Ratio |
|---|---:|---:|---:|---:|---:|---:|
| DINOv2 | TBD | TBD | TBD | TBD | TBD | TBD |
| DINOv2 + dropout | TBD | TBD | TBD | TBD | TBD | TBD |
| Descriptor-only distillation | TBD | TBD | TBD | TBD | TBD | TBD |
| GeoToken | TBD | TBD | TBD | TBD | TBD | TBD |

### 4.5 Ablation Study

We isolate the contribution of each PSRD component.

**Table 4. PSRD ablation.**

| Variant | Spatial QA | Relation Probe | BEV IoU | Robust QA |
|---|---:|---:|---:|---:|
| No LiDAR privileged supervision | TBD | TBD | TBD | TBD |
| Depth-only distillation | TBD | TBD | TBD | TBD |
| BEV-only distillation | TBD | TBD | TBD | TBD |
| Descriptor-only distillation | TBD | TBD | TBD | TBD |
| PSRD without metric distance | TBD | TBD | TBD | TBD |
| PSRD without direction | TBD | TBD | TBD | TBD |
| PSRD without topology | TBD | TBD | TBD | TBD |
| Full PSRD | TBD | TBD | TBD | TBD |

We also evaluate clip length, number of cameras, and adapter capacity:

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

We visualize token affinity maps and compare them with LiDAR-derived spatial relations. GeoToken should produce stronger affinity between tokens that are close in 3D space even when they are far in the image plane or located across adjacent camera views. We also visualize failure cases under heavy occlusion, night scenes, and missing cameras.

Suggested figures:

- **Figure 1:** Overview of GeoToken training and inference.
- **Figure 2:** LiDAR spatial relation graph construction from projected points and BEV cells.
- **Figure 3:** PSRD loss: relation, distance, direction, and topology supervision.
- **Figure 4:** VLM adapter evaluation protocol with frozen VLM.
- **Figure 5:** Qualitative token affinity under clean and degraded inputs.

---

## 5. Discussion

### Why Relations Instead of Dense Depth?

Dense depth supervision improves metric cues, but it does not directly teach a VLM which entities are closer, which regions are connected, or how objects relate across views. Driving decisions often depend on relational geometry rather than per-pixel depth alone. PSRD is designed to supervise those relations explicitly.

### Why Frozen VLM Evaluation?

Full VLM fine-tuning can obscure whether improvements come from better visual tokens or from language model adaptation. By freezing the VLM and training only a small adapter, the evaluation more directly measures the usefulness of the pretrained camera tokens.

### Deployment Cost

GeoToken requires LiDAR only during training. Inference uses only camera video and a lightweight token adapter, making it compatible with camera-only deployment.

---

## 6. Limitations

GeoToken depends on paired camera-LiDAR data during pretraining. Datasets without LiDAR cannot directly provide PSRD supervision. The relation graph also depends on calibration quality and LiDAR visibility; sparse or noisy LiDAR may provide incomplete supervision for distant or heavily occluded regions. Finally, while frozen VLM spatial QA is a strong proxy for VLM usefulness, it does not replace full closed-loop driving evaluation. Future work should integrate GeoToken with closed-loop VLA systems and evaluate planning behavior directly.

---

## 7. Conclusion

We presented GeoToken, a LiDAR-privileged spatial token pretraining framework for camera-only driving VLMs. The core method, Privileged Spatial Relation Distillation, transfers metric, directional, and occupancy-topology relations from LiDAR to camera token interactions. GeoToken is designed to improve spatial reasoning and robustness while requiring only cameras at inference. Through VLM spatial QA, frozen spatial probing, robustness evaluation, and localization retention, this work aims to establish training-time 3D sensing as a practical route toward spatially grounded camera-only driving VLMs.

---

## References

[1] Katrin Renz, Long Chen, Elahe Arani, and Oleg Sinavski. SimLingo: Vision-Only Closed-Loop Autonomous Driving with Language-Action Alignment. CVPR, 2025.

[2] Haiming Zhang et al. VisionPAD: A Vision-Centric Pre-training Paradigm for Autonomous Driving. CVPR, 2025.

[3] SpaceDrive: Infusing Spatial Awareness into VLM-based Autonomous Driving. arXiv, 2025.

[4] Dacheng Liao et al. RoboDriveVLM: A Novel Benchmark and Baseline towards Robust Vision-Language Models for Autonomous Driving. arXiv, 2025.

[5] DriveLM: Driving with Graph Visual Question Answering. ECCV, 2024.

[6] Xiaoyu Tian et al. DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models. CoRL, 2025.

[7] SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities. CVPR, 2024.

[8] VLM-3R: Vision-Language Models Augmented with Instruction-Aligned 3D Reconstruction. CVPR, 2026.

[9] DRIVESPATIAL: A Benchmark for Spatiotemporal Intelligence in VLMs for Autonomous Driving. arXiv, 2026.

[10] nuScenes: A multimodal dataset for autonomous driving. CVPR, 2020.

[11] KITTI-360: A novel dataset and benchmarks for urban scene understanding in 2D and 3D. TPAMI, 2022.
