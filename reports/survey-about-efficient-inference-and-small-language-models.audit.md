# Citation Audit

1. **Claim:** "Small Language Models (SLMs) can outperform larger models in specialized, latency-sensitive environments, challenging the 'larger is better' paradigm."
   - **Verdict:** SUPPORTED
   - **Evidence:** The paper evaluates 9 models on a deployed agentic system and finds that "larger models are not uniformly better" and that "capability is not ordered the same way at every site."
   - **URL:** https://arxiv.org/html/2610.09021

2. **Claim:** "Quasar introduces quantized verification techniques to alleviate this memory pressure."
   - **Verdict:** SUPPORTED
   - **Evidence:** The abstract states that Quasar is "a novel, training-free framework designed to overcome this 'memory wall' by employing low-bit quantization specifically for the verification stage."
   - **URL:** https://arxiv.org/abs/2603.01399

3. **Claim:** "PhoneLM demonstrate the efficacy of architecture searching tailored for on-device deployment."
   - **Verdict:** SUPPORTED
   - **Evidence:** The paper states, "we develop a simple yet effective principle for SLM design: architecture searching for (near-)optimal runtime efficiency before pre-training."
   - **URL:** https://huggingface.co/papers/2411.05046

4. **Claim:** "Modular approaches like Mixture-of-Task-Adapters (MoTA) allow models to handle multiple tasks efficiently by activating only relevant parameters."
   - **Verdict:** SUPPORTED
   - **Evidence:** The abstract describes ALTER as a system using "Mixture-of-Task-Adapters to efficiently handle multiple NLP tasks while minimizing computational cost."
   - **URL:** https://huggingface.co/papers/2309.11042

5. **Claim:** "A separate Microsoft case study examined natural-language interaction with an internal cloud supply-chain application and reported that small models could outperform larger models in accuracy and running time in that setting."
   - **Verdict:** SUPPORTED
   - **Evidence:** The arXiv abstract reports that small models can outperform much larger ones in both accuracy and running time for a particular internal Microsoft application used in cloud supply-chain fulfilment. It describes a case study, not a general deployment comparison.
   - **URL:** https://arxiv.org/pdf/2405.20347
