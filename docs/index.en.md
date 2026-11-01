# Use Token to Learn · Visual LLM Knowledge Base

> Learn large language models by playing with them: every topic comes with **a write-up, figures, an interactive playground and runnable PyTorch code**.
>
> 🌐 [中文](index.md)

## How to learn

Each chapter follows the same rhythm:

1. **🎮 Play first**: drag sliders and click buttons to build intuition
2. **📖 Then read**: intuition → pictures → math → code → misconceptions → self-check
3. **🧑‍💻 Then run the code**: a minimal PyTorch implementation checked against the built-in one

## Chapters

| # | Topic | Playground | Status |
| --- | --- | --- | --- |
| 01 | [Tokenization](01-tokenization/README.md) | [Tokenizer playground](01-tokenization/interactive/tokenizer.html?lang=en) | ✅ done |
| 02 | [Embedding](02-embedding/README.md) | [Embedding playground](02-embedding/interactive/embedding.html?lang=en) | ✅ done |
| 03 | [Positional encoding](03-positional-encoding/README.md) | [Positional encoding playground](03-positional-encoding/interactive/positional.html?lang=en) | ✅ done |
| 04 | [Attention](04-attention/README.md) | [Attention playground](04-attention/interactive/attention.html?lang=en) | ✅ done |
| 05 | [Transformer block](05-transformer/README.md) | [Transformer playground](05-transformer/interactive/transformer.html?lang=en) | ✅ done |
| 06 | [Pre-training](06-pretraining/README.md) | [Pre-training playground](06-pretraining/interactive/pretraining.html?lang=en) | ✅ done |
| 07 | [Scaling laws](07-scaling-laws/README.md) | [Scaling laws playground](07-scaling-laws/interactive/scaling.html?lang=en) | ✅ done |
| 08 | [Fine-tuning](08-finetuning/README.md) | [Fine-tuning playground](08-finetuning/interactive/finetune.html?lang=en) | ✅ done |
| 09 | [Alignment](09-alignment/README.md) | [Alignment playground](09-alignment/interactive/alignment.html?lang=en) | ✅ done |
| 10 | [Reasoning models and test-time compute](10-reasoning/README.md) | [Reasoning playground](10-reasoning/interactive/reasoning.html?lang=en) | ✅ done |
| 11 | [Decoding and sampling](11-decoding/README.md) | [Decoding playground](11-decoding/interactive/decoding.html?lang=en) | ✅ done |
| 12 | [Inference optimization](12-inference-optimization/README.md) | [Inference optimization playground](12-inference-optimization/interactive/inference.html?lang=en) | ✅ done |
| 13 | [Serving systems](13-serving/README.md) | [Scheduling and memory playground](13-serving/interactive/serving.html?lang=en) | ✅ done |
| 14 | [Long context](14-long-context/README.md) | [RoPE extrapolation and attention patterns](14-long-context/interactive/long_context.html?lang=en) | ✅ done |
| 15 | [Mixture of experts](15-moe/README.md) | [Routing and load-balancing playground](15-moe/interactive/moe.html?lang=en) | ✅ done |
| 16 | [Multimodal models](16-multimodal/README.md) | [Patches, CLIP and zero-shot, VLM captions playground](16-multimodal/interactive/multimodal.html?lang=en) | ✅ done |
| 17 | [Code models](17-code-models/README.md) | [FIM, pass@k and test-based selection playground](17-code-models/interactive/code_models.html?lang=en) | ✅ done |
| 18 | [Prompt engineering](18-prompt-engineering/README.md) | [Prompt bench: two models answer live](18-prompt-engineering/interactive/prompting.html?lang=en) | ✅ done |
| 19 | [RAG and vector databases](19-rag/README.md) | [Retrieval bench and HNSW / IVF search playground](19-rag/interactive/rag.html?lang=en) | ✅ done |
| 20 | [Knowledge graphs and graph databases](20-knowledge-graphs/README.md) | [Extraction, Cypher queries and multi-hop retrieval playground](20-knowledge-graphs/interactive/knowledge_graph.html?lang=en) | ✅ done |
| 21 | [Agents and the agent loop](21-agents/README.md) | [Step-through agent loop with LangGraph nodes](21-agents/interactive/agents.html?lang=en) | ✅ done |
| 22 | [MCP, Skills and SOPs](22-mcp-skills-sop/README.md) | [MCP messages, skill progressive disclosure and an SOP pipeline](22-mcp-skills-sop/interactive/mcp.html?lang=en) | ✅ done |
| 23 | [Evaluation](23-evaluation/README.md) | [Perplexity, leaderboard error bars, LLM-judge bias and contamination detection](23-evaluation/interactive/evaluation.html?lang=en) | ✅ done |
| 24 | [Safety and interpretability](24-safety-interpretability/README.md) | [Probes and steering, superposition and an SAE feature browser, hallucination grading and jailbreaks](24-safety-interpretability/interactive/safety.html?lang=en) | ✅ done |

## Frontier Q&A

Specific, often very recent questions, one per page, with sources and a check date:

- [How are reasoning effort levels (low / medium / high / xhigh) set?](qa/reasoning-effort.en.md)
- [What is DeepSeek's open-source DeepInfer, and why does it matter?](qa/deepinfer.en.md)
- [How is LLM compute measured, where does it go, and how does it compare with RNNs, CNNs and KNN?](qa/compute.en.md) (with a [compute calculator](qa/interactive/compute.html?lang=en))
- [How do CNNs, RNNs and KNN work, and how are they different from LLMs?](qa/classic-models.en.md) (with a [playground](qa/interactive/classic-models.html?lang=en))
- [How do I make LLM outputs reliable, and stop the model making things up?](qa/reliability.en.md) (with a [playground](qa/interactive/reliability.html?lang=en))
- [All Q&A](qa/index.en.md) · [Glossary](glossary.en.md)

See [PLAN.md](https://github.com/BaiYuBY/use_token_to_learn/blob/main/PLAN.md) in the repository for the full roadmap.
