# Reinforcement Learning for Large Language Model Reasoning: A Comprehensive Survey

## TL;DR
- Reinforcement learning has evolved from offline preference optimization (DPO) and heuristic alignment (RLHF via PPO [1]) to advanced reasoning paradigms powered by Group Relative Policy Optimization (GRPO) [2] and verifiable rewards (RLVR) [3].
- Multi-step reasoning challenges are addressed through granular credit assignment using Process Reward Models (PRMs) [4], tree-guided search (ReST-MCTS*) [5], and reward-guided self-training frameworks [6].
- Recent open-source reasoning milestones demonstrate distinct training pathways: DeepSeek-R1-Zero uses pure reinforcement learning on verifiable domains without initial SFT to spontaneously elicit advanced self-reflection and test-time scaling behaviors, whereas DeepSeek-R1 incorporates cold-start data and staged training [2].
- Key technical tradeoffs center on balancing exploration diversity (ROSE) [7], mitigating reward hacking in subjective tasks, and controlling token inefficiency via early-exit strategies (S-GRPO) [8].

## Background
Reinforcement learning (RL) for large language model (LLM) reasoning refers to the computational paradigm of optimizing language generation policies to maximize task correctness, logical coherence, and step-by-step verification over multi-turn or multi-step trajectories. Historically, alignment and generation were guided by Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF), formalized via Proximal Policy Optimization (PPO) against a learned neural reward model [1]. While effective for general instruction-following and safety, standard response-level RLHF struggles with complex multi-step reasoning tasks such as advanced mathematics and computer programming, where a single logical flaw early in the chain invalidates the final output.

To address these credit assignment bottlenecks, research expanded into offline preference optimization alternatives such as Direct Preference Optimization (DPO) [9], step-wise preference optimization (Step-DPO) [10], and process supervision [4]. Foundational work established that aligning models with human or verifiable preferences requires careful regulation of policy drift through KL-divergence penalties [1] or closed-form preference loss [9]. Recent breakthroughs have shifted focus toward Reinforcement Learning with Verifiable Rewards (RLVR) [3], where rule-based parsers (e.g., executing code or checking numerical answers) provide exact ground-truth feedback. While RLVR mitigates neural reward model vulnerabilities on verifiable tasks, rule-based verification alone does not eliminate reward hacking when reward functions can be exploited by degenerate generation patterns [2].

## Foundations: From Preference Alignment to Step-Wise Supervision
The architectural foundation of LLM reinforcement learning includes the classical three-step RLHF pipeline: SFT, Reward Modeling (RM) via Bradley-Terry pairwise preference regression, and policy optimization via PPO [1]. To avoid policy collapse, PPO incorporates a per-token KL-divergence penalty against the initial reference policy [1]. However, maintaining actor, critic, reward, and reference models simultaneously introduces severe computational overhead. This motivated Direct Preference Optimization (DPO), which serves as an offline preference optimization alternative rather than an active RL training loop, reparameterizing the reward function analytically to optimize language model policies directly using binary cross-entropy loss over offline preference pairs [9].

While response-level alignment succeeds for general generation, complex reasoning requires granular credit assignment. Step-DPO extends preference optimization by targeting individual reasoning steps in Chain-of-Thought (CoT) trajectories, penalizing exact error locations rather than entire completions [10]. Complementing this, reward models have evolved from Outcome Reward Models (ORMs)—which evaluate only final answers—to Process Reward Models (PRMs) that score intermediate reasoning steps [4]. Furthermore, Reward Reasoning Models (RRMs) leverage test-time compute scaling, allowing the reward model itself to deliberate via internal chains of thought before scoring candidate steps [11].

## Advanced Methods: Process Rewards, Tree Search, and Policy Optimization
As reasoning tasks grow more intricate, simple generation and scoring become insufficient, driving the adoption of search-augmented RL and theoretical unification between outcome and process supervision. ReST-MCTS* integrates process reward guidance with Monte Carlo Tree Search to collect high-quality reasoning traces and per-step values without manual step annotation, using rule-based final verifiers to infer process values [5]. Similarly, Reinforced Efficient Reasoning via Semantically Diverse Exploration (ROSE) addresses limited search diversity in tree-based rollouts to improve segment-level credit assignment [7], while ReST-RL unifies self-training policy updates with value-guided search [6].

A major theoretical development demonstrates that Group Relative Policy Optimization (GRPO) with outcome rewards is mathematically equivalent under mild assumptions to a PRM-aware RL objective equipped with an implicit Monte Carlo PRM derived from overlapping trajectory prefixes [12]. This insight explains how outcome-based RL achieves fine-grained credit assignment without explicit step-level supervision. Supporting this, theoretical analyses on supervision complexity establish that, under standard data coverage assumptions and trajectory measure lemmas, outcome supervision is no more statistically difficult than process supervision up to polynomial horizon factors [13].

## Algorithmic Innovations in Open-Source Reasoning Paradigms
The release of DeepSeek-R1 highlighted distinct training paradigms: DeepSeek-R1-Zero applies pure reinforcement learning on verifiable domains without initial supervised fine-tuning, whereas DeepSeek-R1 utilizes cold-start data and multi-stage training to incentivize advanced reasoning capabilities [2]. Both operate on RLVR with rule-based verification (checking mathematical equality and code compilation) using GRPO, which foregoes the traditional standalone critic model and estimates score baselines directly from group-sampled outputs [2]. This eliminates massive memory bottlenecks and enables scalable chain-of-thought generation.

```
+-------------------------------------------------------------------+
|                  DeepSeek-R1 RLVR & GRPO Pipeline                 |
+-------------------------------------------------------------------+
| 1. Prompt Input (Math / Code / Logic)                             |
|    v                                                              |
| 2. Group Generation (Model samples G distinct CoT trajectories)   |
|    v                                                              |
| 3. Rule-Based Verification (Execution parser checks final answer) |
|    v                                                              |
| 4. Group Normalization (Compute mean/std score baseline in group) |
|    v                                                              |
| 5. Policy Update (GRPO loss update without separate critic model) |
+-------------------------------------------------------------------+
```

Despite their success, these large-scale RL reasoning models introduce specific training dynamics and failure modes. Discriminative Constrained Optimization (DisCO) addresses training instabilities and difficulty bias in standard GRPO by enforcing entropy stability during large-scale optimization [14]. Additionally, S-GRPO introduces serial-group decaying rewards to encourage early exits during generation, curbing token inefficiency and excessive generation length on straightforward problems [8]. Investigation into the geometric properties of RLVR using trainable activation vectors reveals that effective reasoning manifolds in transformer activation space can be highly compressed, though extreme compression risks intervention anomalies [3].

| Method / Framework | Primary Mechanism | Supervision Type | Key Advantage | Main Limitation |
|---|---|---|---|---|
| **RLHF (PPO)** [1] | Actor-Critic with KL penalty | Human pairwise preference | Stable general alignment; avoids catastrophic drift | High memory overhead; critic model complexity |
| **DPO** [9] | Offline preference optimization | Offline preference pairs | Eliminates explicit RM and RL training loops | Whole-response granularity; offline distribution shift |
| **Step-DPO** [10] | Step-wise preference loss | Step-level annotations / rules | Precise error localization in long CoT | Requires step-level verification data or parsers |
| **ReST-MCTS\*** [5] | MCTS tree search + PRM guidance | Rule-based final verification | High-quality trajectory collection & value guidance | Computationally intensive search overhead |
| **GRPO** [2] | Group-normalized score baseline | Verifiable rules (RLVR) | Eliminates critic model; scales long CoT efficiently | Vulnerable to reward hacking and group variance |
| **S-GRPO** [8] | Serial-group decaying reward | Verifiable rules + early exit | Reduces token inefficiency & overthinking | Risk of truncating necessary verification steps |

## Trends and open problems
The landscape of RL for LLM reasoning exhibits several prominent trends alongside persistent open challenges:
1. **Shift Toward Verifiable Rewards (RLVR):** The success of rule-based verification for math and coding has shifted momentum toward programmatic verification, providing precise ground-truth signals in deterministic domains [2].
2. **Test-Time Compute and Dynamic Scaling:** Modern reasoning models allocate variable compute budgets dynamically (generating extensive internal chains of thought for complex queries) [2], though mitigating excessive token expenditure on trivial inputs remains an active engineering target (S-GRPO) [8].
3. **Generalization Beyond Verifiable Domains:** While RLVR excels in formal logic, mathematics, and programming, scaling pure RL or reward models to subjective, open-ended, or multi-agent domains remains constrained by reward hacking and alignment tax [2].
4. **Theoretical Unification of Process and Outcome Supervision:** Recent proofs showing that group outcome optimization implicitly performs Monte Carlo process supervision under mild assumptions [12] bridge the gap between expensive step annotation and coarse outcome rewards, pointing toward more unified training algorithms.

## References
[1] Training language models to follow instructions with human feedback. arxiv. https://arxiv.org/abs/2203.02155 (2022-03-04)
[2] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. hf-search. https://huggingface.co/papers/2501.12948 (2025-01-22)
[3] Learning to Steer, Steering to See: Unveiling the Geometry of RLVR in Large Language Models via Trainable Vectors. hf-daily. https://huggingface.co/papers/2609.34344 (2026-09-28)
[4] Enhancing Large Language Model Reasoning with Reward Models: An Analytical Survey. hf-search. https://huggingface.co/papers/2510.01925 (2025-10-02)
[5] ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search. web. https://proceedings.neurips.cc/paper_files/paper/2024/file/76ec4dc30e9faaf0e4b6093eaa377218-Paper-Conference.pdf (2024)
[6] ReST-RL: Reinforcing LLM Reasoning through Unified Self-Training and Value-Guided Search. arxiv. https://arxiv.org/abs/2508.19576 (2025-08-27)
[7] Reinforced Efficient Reasoning via Semantically Diverse Exploration. arxiv. https://arxiv.org/abs/2601.05053 (2026-01-08)
[8] S-GRPO: Early Exit via Reinforcement Learning in Reasoning Models. hf-search. https://huggingface.co/papers/2505.07686 (2025-05-12)
[9] Direct Preference Optimization: Your Language Model is Secretly a Reward Model. arxiv. https://arxiv.org/abs/2305.18290 (2023-05-29)
[10] Step-DPO: Step-wise Preference Optimization for Long-chain Reasoning of LLMs. arxiv. https://arxiv.org/abs/2406.18629 (2024-06-26)
[11] Reward Reasoning Model. hf-search. https://huggingface.co/papers/2505.14674 (2025-05-20)
[12] GRPO is Secretly a Process Reward Model. web. https://arxiv.org/abs/2509.21154 (2025-09-25)
[13] Do We Need to Verify Step by Step? Rethinking Process Supervision from a Theoretical Perspective. web. https://raw.githubusercontent.com/mlresearch/v267/main/assets/jia25f/jia25f.pdf (2025)
[14] DisCO: Reinforcing Large Reasoning Models with Discriminative Constrained Optimization. hf-search. https://huggingface.co/papers/2505.12366 (2025-05-18)
