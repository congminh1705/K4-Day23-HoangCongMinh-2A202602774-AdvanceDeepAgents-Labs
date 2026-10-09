# Survey of World Models: Architectures, Scaling Paradigms, and Open Challenges

## TL;DR
- World models learn internal simulation dynamics of environments to empower reinforcement learning, planning, and interactive video generation without requiring online physical interaction [1], [2], [3].
- Modern architectures span discrete latent recurrent models (DreamerV3) [4], value-equivalent planning models (MuZero) [5], unsupervised video foundation models (Genie) [6], and real-time diffusion transformers (GameNGen, DIAMOND, Oasis) [2], [7], [8].
- Non-generative joint-embedding predictive architectures (V-JEPA) bypass pixel-level reconstruction for robust feature-space physical reasoning and high sample/label efficiency [9].
- Critical bottlenecks include multi-step hallucination, kinematic-only imagination lacking true physical force/friction dynamics, and heavy computational costs [10], [11], [3].

## Background
World models refer to computational systems designed to capture and simulate the spatial, temporal, and dynamic transitions of an environment. The paradigm was formalized in early neural architectures by Ha and Schmidhuber (2018), who demonstrated that agents can learn unsupervised compressed spatial-temporal representations of environments (using VAEs and RNNs) and train policies entirely within their own hallucinated "dreams" before zero-shot transfer to real environments [1]. This decouples policy learning from costly real-world interactions.

Subsequent milestones refined how internal models are leveraged. MuZero (2019/2020) eliminated the need to reconstruct high-dimensional observations, introducing recursive models that predict action-selection policies, values, and rewards directly relevant for tree-based planning [5]. DreamerV3 (2023) established a unified model-based reinforcement learning framework capable of mastering over 150 diverse domains—ranging from continuous control to sparse-reward 3D environments like Minecraft—using a fixed hyperparameter set [4]. More recently, foundational video generation models such as Genie (2024) demonstrated unsupervised training from unlabelled internet videos to generate interactive, action-controllable virtual worlds [6].

## Foundations and Methodological Paradigms
World model methodologies diverge primarily in how they represent environmental state transitions and how those representations serve downstream control. Three dominant foundational paradigms have emerged: recurrent state-space models (RSSMs), value-equivalent planning models, and latent feature predictors.

RSSMs, epitomized by the Dreamer family, couple deterministic recurrent neural networks with stochastic categorical states to model Partially Observable Markov Decision Processes (POMDPs). By unrolling latent state transitions conditioned on agent actions, agents perform imagination rollouts where both policy and value networks are optimized entirely in latent space [4]. In contrast, value-equivalent models like MuZero bypass pixel reconstruction entirely, optimizing latent representations exclusively to predict value functions, reward signals, and policy distributions required for Monte Carlo Tree Search [5].

A third major paradigm is represented by non-generative joint-embedding predictive architectures such as V-JEPA (2024). Instead of reconstructing pixels or predicting text tokens—which wastes compute on unpredictable high-frequency textures—V-JEPA utilizes Siamese Vision Transformers to predict masked spatio-temporal regions entirely within abstract latent feature space [9]. This yields high label and sample efficiency and robust representations for downstream evaluation benchmarks [9].

## Modern Architectures and Video Generation Scaling
The boundary between generative video models and interactive simulators has blurred dramatically with the adoption of diffusion models and Vision Transformers (ViTs). GameNGen (2024) demonstrated that neural diffusion models can serve as real-time game engines, interactively simulating complex environments like DOOM at 20 frames per second on a single TPU [2]. By introducing noise-conditioning augmentations during training to prevent autoregressive drift and fine-tuning decoder latent losses, GameNGen maintained visual stability over long trajectories [2].

Similarly, DIAMOND (2024) proved that diffusion-based world models significantly outperform discrete latent models in Partially Observable MDPs by preserving fine visual details, achieving a new state-of-the-art mean human normalized score of 1.46 on Atari 100k [7]. Expanding this trajectory, Oasis (2024) introduced a real-time open-world interactive model (simulating Minecraft) built entirely on Vision Transformers and Diffusion Transformers (DiT) operating under Diffusion Forcing without an explicit physics engine [8].

To address spatial grounding and physical plausibility, recent architectures have incorporated multi-task geometric supervision. XGenAct (2026) unifies RGB predictions with metric depth, surface normals, and robot actions, bridging video generation with spatial robot manipulation [12]. Likewise, DreamTrue (2026) addresses cross-embodiment robot prediction by employing offline geometric calibration and counterfactual post-training to eliminate biases toward successful outcomes [13].

| Model / Framework | Architecture Type | Primary Modality | Key Advantage | Major Limitation |
| :--- | :--- | :--- | :--- | :--- |
| **DreamerV3** [4] | RSSM / Latent Dynamics | Continuous Control / 3D Sims | Unified hyperparameters across 150+ diverse domains | Vulnerable to latent drift in out-of-distribution states |
| **MuZero** [5] | Recursive Tree-Search Model | Board Games / Atari | Avoids pixel reconstruction; optimized for planning | Computationally intensive tree search; typically applied to discrete action domains |
| **Genie** [6] | Spatiotemporal Tokenizer + Autoregressive Transformer | Unlabeled Internet Videos | Unsupervised learning of controllable latent actions | Massive compute scale (11B params); unaligned actions |
| **GameNGen** [2] | Augmented Diffusion Model | Interactive Video (DOOM) | Real-time 20 FPS interactive simulation | Rare state hallucination over extended horizons |
| **V-JEPA** [9] | Joint-Embedding Predictive ViT | Video Feature Prediction | High sample efficiency without pixel reconstruction | Non-generative; typically evaluated as a feature extractor requiring task-specific heads or decoders for control |
| **Oasis** [8] | Vision Transformer + DiT (Diffusion Forcing) | Open-World Simulation (Minecraft) | Scalable Transformer architecture without physics engine | Physics rule degradation in unvisited map regions |

## Applications across Robotics and Autonomous Driving
World models have become central to safety-critical domains where real-world trial-and-error is prohibitively dangerous or data-expensive. In autonomous driving, comprehensive surveys highlight how generative world models integrate heterogeneous sensor streams—including LiDAR, radar, and cameras—into unified 4D occupancy grids (e.g., OccWorld, OccSora, TrafficBots) [14]. These models simulate future traffic dynamics, evaluate multi-agent collision risks, and synthesize rare safety-critical edge cases, enabling end-to-end planning that bypasses error-prone modular pipelines [14].

In robotic manipulation and locomotion, simulation distillation frameworks (SimDist) pretrain world models inside physics engines by systematically injecting erroneous actions into rollouts, exposing agents to failures and recoveries [15]. During deployment, the perception and reward modules are frozen while the latent dynamics model adapts rapidly to real-world sensory feedback [15].

## Trends and open problems
Despite remarkable progress, world models face profound scientific hurdles regarding reliability and generalization. Recent diagnostic studies categorize long-horizon failures into distinct failure modes. Hallucinations in generative world models are highly predictable through metrics like tokenizer round-trip residuals and flow instability, yet remain dangerous because silent rollout errors feed directly into downstream controllers [10].

Furthermore, diagnostic evaluations such as the imagined Kinematic-Consistency Error (iKCE) reveal that many world models imagine *kinematically* rather than *dynamically*—generating plausible visual motions while failing to respect underlying physical forces, mass, or friction thresholds [11]. Comprehensive surveys also note that perception metrics (FID, FVD, PSNR) correlate poorly with downstream causal validity, with benchmarks like WorldBench indicating that state-of-the-art models achieve roughly 45% foreground mIoU on physical reasoning probes [3]. Resolving compounding errors, eliminating multi-step drift, and enforcing rigorous physical constraints remain primary challenges for future research.

## References
[1] World Models. arxiv. https://arxiv.org/abs/1803.10122 (2018-03-27)
[2] Diffusion Models Are Real-Time Game Engines. arxiv. https://arxiv.org/abs/2408.14837 (2024-08-27)
[3] World Models: A Comprehensive Survey of Architectures, Methodologies, Reasoning Paradigms, and Applications. hf-search. https://huggingface.co/papers/2606.00133 (2026-06-02)
[4] Mastering Diverse Domains through World Models. arxiv. https://arxiv.org/abs/2301.04104 (2023-01-10)
[5] Mastering Atari, Go, chess and shogi by planning with a learned model. arxiv. https://arxiv.org/abs/1911.08265 (2019-11-18)
[6] Genie: Generative Interactive Environments. arxiv. https://arxiv.org/abs/2402.15391 (2024-02-23)
[7] Diffusion for World Modeling: Visual Details Matter in Atari. arxiv. https://arxiv.org/abs/2405.12399 (2024-05-20)
[8] Oasis: A Universe in a Transformer. web. https://oasis-model.github.io/ (2024-10-31)
[9] Revisiting Feature Prediction for Learning Visual Representations from Video. arxiv. https://arxiv.org/abs/2404.08471 (2024-04-12)
[10] Hallucination in World Models is Predictable and Preventable. arxiv. https://arxiv.org/abs/2606.27326 (2026-06-25)
[11] Imagined Rollouts Are Kinematic, Not Dynamic: A Diagnosis of Long-Horizon World-Model Failure. arxiv. https://arxiv.org/abs/2607.05966 (2026-07-07)
[12] XGenAct: Geometry-Enhanced World Action Models through Cross-Task Generation. arxiv. https://arxiv.org/abs/2610.03516 (2026-10-02)
[13] DreamTrue: Action-Faithful Robot World Model with Counterfactual Post-Training. arxiv. https://arxiv.org/abs/2610.12468 (2026-10-08)
[14] A Comprehensive Survey of World Models for Autonomous Driving. arxiv. https://arxiv.org/abs/2501.11260 (2025-01-19)
[15] Simulation Distillation: Pretraining World Models in Simulation for Rapid Real-World Adaptation. web. https://roboticsproceedings.org/rss22/p017.pdf (2026-07-13)
