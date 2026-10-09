# Citation Audit Report

For each of the 5 claims:

- Claim number: [1]
- Verbatim claim from report: "Historically, alignment and generation were guided by Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF), formalized via Proximal Policy Optimization (PPO) against a learned neural reward model [1]."
- Source URL: https://arxiv.org/abs/2203.02155
- Audit status: SUPPORTED
- Evidence & notes: The source paper ("Training language models to follow instructions with human feedback", Ouyang et al., 2022) introduces InstructGPT, which formalizes the alignment pipeline using supervised fine-tuning followed by reinforcement learning from human feedback (RLHF) optimized via PPO against a learned reward model trained on human preference data.

- Claim number: [2]
- Verbatim claim from report: "To address these credit assignment bottlenecks, research expanded into offline preference optimization alternatives such as Direct Preference Optimization (DPO) [2], step-wise preference optimization (Step-DPO) [10], and process supervision [5]."
- Source URL: https://arxiv.org/abs/2305.18290
- Audit status: SUPPORTED
- Evidence & notes: The source paper ("Direct Preference Optimization: Your Language Model is Secretly a Reward Model", Rafailov et al., 2023) introduces Direct Preference Optimization (DPO) as an offline preference optimization alternative that optimizes language model policies directly on preference data without an explicit separate reward model or online reinforcement learning loop.

- Claim number: [3]
- Verbatim claim from report: "The release of DeepSeek-R1 highlighted distinct training paradigms: DeepSeek-R1-Zero applies pure reinforcement learning on verifiable domains without initial supervised fine-tuning, whereas DeepSeek-R1 utilizes cold-start data and multi-stage training to incentivize advanced reasoning capabilities [3]."
- Source URL: https://huggingface.co/papers/2501.12948
- Audit status: SUPPORTED
- Evidence & notes: The source paper ("DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", DeepSeek-AI, 2025) explicitly distinguishes DeepSeek-R1-Zero (which relies on pure RL without initial SFT) from DeepSeek-R1 (which incorporates cold-start data and staged training).

- Claim number: [10] (cited as [10] in report for Step-DPO)
- Verbatim claim from report: "Step-DPO extends preference optimization by targeting individual reasoning steps in Chain-of-Thought (CoT) trajectories, penalizing exact error locations rather than entire completions [10]."
- Source URL: https://arxiv.org/abs/2406.18629
- Audit status: SUPPORTED
- Evidence & notes: The source paper ("Step-DPO: Step-wise Preference Optimization for Long-chain Reasoning of LLMs", Lai et al., 2024) presents Step-DPO, which treats individual reasoning steps in chain-of-thought trajectories as units for preference optimization rather than evaluating answers holistically, allowing precise targeting of errors in long-chain mathematical reasoning.

- Claim number: [12] (cited as [12] in report for GRPO as a PRM)
- Verbatim claim from report: "A major theoretical development demonstrates that Group Relative Policy Optimization (GRPO) with outcome rewards is mathematically equivalent under mild assumptions to a PRM-aware RL objective equipped with an implicit Monte Carlo PRM derived from overlapping trajectory prefixes [12]."
- Source URL: https://arxiv.org/abs/2509.21154
- Audit status: SUPPORTED
- Evidence & notes: The source paper ("GRPO is Secretly a Process Reward Model", Sullivan & Koller, 2025) provides theoretical proof that the Group Relative Policy Optimization (GRPO) algorithm equipped with an outcome reward model is equivalent under mild assumptions to a PRM-aware RL objective equipped with a Monte-Carlo-based PRM derived from overlapping trajectory prefixes.
