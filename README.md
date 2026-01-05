<p align="center">
  <img src="https://img.shields.io/badge/LanguageForge-Language%20Models%20Research%20Lab-6A5ACD?style=for-the-badge">
</p>

<h1 align="center">🔡 LanguageForge</h1>

<p align="center">
  <i>
    A self-directed research laboratory dedicated to understanding how
    <b>Language Models</b> are designed, trained, scaled, and evaluated —
    from statistical foundations to modern Large Language Models.
  </i>
</p>

<!-- Badges -->
<p align="center">
  <img src="https://img.shields.io/badge/Language%20Models-Core-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Transformers-Architecture-purple?style=for-the-badge">
  <img src="https://img.shields.io/badge/Tokenization-BPE%20%7C%20WordPiece%20%7C%20Unigram-1f6feb?style=for-the-badge">
  <img src="https://img.shields.io/badge/Embeddings-Word%20%7C%20Sentence-6f42c1?style=for-the-badge">
  <img src="https://img.shields.io/badge/Training-Pretraining%20%7C%20SFT-0ea5e9?style=for-the-badge">
  <img src="https://img.shields.io/badge/Alignment-RLHF%20(Conceptual)-22c55e?style=for-the-badge">
  <img src="https://img.shields.io/badge/Evaluation-Reliability-red?style=for-the-badge">
</p>

---

## 🔡 About LanguageForge

**LanguageForge** is a focused **research and learning archive** documenting my
deep dive into **Language Models and Large Language Models (LLMs)**.

The emphasis of this repository is **internal understanding**:

- how text becomes tokens  
- how meaning emerges from embeddings  
- how attention enables sequence modeling  
- how large-scale training shapes behavior  
- how alignment and evaluation affect reliability  

This repository deliberately **excludes system-level applications**
(e.g., RAG, agents, prompt engineering) to preserve **conceptual depth and clarity**.

> *LanguageForge is where I study how language models work — not how they are used.*

---

## 🗂️ Table of Contents

| No. | Module | Focus Area | Key Topics | Status |
|:--:|:--------|:-----------|:-----------|:------|
| 1 | [Foundations of Language Models](#-module-1-foundations-of-language-models) | Core concepts | Tokens, embeddings, scaling laws | 🟢 Active |
| 2 | [Sequence Modeling](#-module-2-sequence-modeling) | Pre-transformer models | n-grams, RNNs, LSTMs | 🟢 Active |
| 3 | [Transformer Architecture](#-module-3-transformer-architecture) | Core LLM architecture | Attention, RoPE, decoder-only | 🟡 In Progress |
| 4 | [Training Language Models](#-module-4-training-language-models) | How LMs learn | Objectives, datasets, scaling | 🔜 Upcoming |
| 5 | [Fine-Tuning & Alignment](#-module-5-fine-tuning--alignment) | Model adaptation | SFT, instruction tuning, RLHF | 🔜 Upcoming |
| 6 | [Parameter-Efficient Fine-Tuning](#-module-6-parameter-efficient-fine-tuning) | Efficient adaptation | LoRA, QLoRA | 🔜 Upcoming |
| 7 | [Inference & Efficiency](#-module-7-inference--efficiency) | Runtime behavior | KV cache, context windows | 🔜 Upcoming |
| 8 | [Evaluation & Reliability](#-module-8-evaluation--reliability) | Model assessment | Hallucinations, metrics | 🧾 Updating |

---

## 🔹 Module 1: Foundations of Language Models

This module introduces the **statistical and representational foundations**
behind language modeling.

**Topics Covered**
- What is a Language Model?
- Language Models vs Foundation Models
- Probability of sequences
- Tokens vs words
- Vocabulary construction
- Scaling intuition (parameters, data, compute)

📓 `Notebooks/00_Foundations/language_model_basics.ipynb`  
📘 `Docs/what_is_a_language_model.md`

---

## 🔹 Module 2: Sequence Modeling

This module studies **pre-transformer approaches** to modeling sequences and
their limitations.

**Topics Covered**
- N-gram language models
- Markov assumptions
- RNNs and vanishing gradients
- LSTMs and GRUs
- Why recurrence failed at scale

📓 `Notebooks/01_Sequence_Models/rnn_lstm_language_models.ipynb`  
📘 `Docs/sequence_modeling_notes.md`

---

## 🔹 Module 3: Transformer Architecture

This module explains the **Transformer** — the backbone of modern LLMs.

**Topics Covered**
- Self-attention mechanism
- Multi-head attention
- Positional encoding (Sinusoidal, RoPE, ALiBi)
- Encoder vs decoder architectures
- Decoder-only LLMs (GPT-style)
- BERT vs GPT (objective & behavior)

📓 `Notebooks/02_Transformers/attention_from_scratch.ipynb`  
📘 `Docs/transformer_architecture.md`

---

## 🔹 Module 4: Training Language Models

This module focuses on **how language models are trained at scale**.

**Topics Covered**
- Autoregressive language modeling
- Masked language modeling
- Causal language models
- Training data pipelines
- Sliding window sampling
- Scaling laws (conceptual)

📓 `Notebooks/03_Training/language_model_training.ipynb`  
📘 `Docs/training_language_models.md`

---

## 🔹 Module 5: Fine-Tuning & Alignment

This module studies how pretrained language models are adapted.

**Topics Covered**
- Supervised fine-tuning (SFT)
- Instruction tuning
- Alignment objectives
- RLHF (high-level pipeline)
- Limitations of alignment

📓 `Notebooks/04_FineTuning/sft_alignment_overview.ipynb`  
📘 `Docs/alignment_notes.md`

---

## 🔹 Module 6: Parameter-Efficient Fine-Tuning

This module explores **efficient adaptation methods** for large models.

**Topics Covered**
- Why full fine-tuning is expensive
- Low-Rank Adaptation (LoRA)
- Quantized LoRA (QLoRA)
- Memory–performance tradeoffs

📓 `Notebooks/05_PEFT/lora_qlora_comparison.ipynb`  
📘 `Docs/peft_notes.md`

---

## 🔹 Module 7: Inference & Efficiency

This module examines **runtime behavior** of language models.

**Topics Covered**
- Context windows
- KV cache intuition
- Throughput vs latency
- Long-context limitations
- Inference-time failures

📓 `Notebooks/06_Inference/inference_efficiency.ipynb`  
📘 `Docs/inference_notes.md`

---

## 🔹 Module 8: Evaluation & Reliability

This module focuses on **evaluating language models beyond accuracy**.

**Topics Covered**
- Intrinsic vs extrinsic evaluation
- Perplexity and its limitations
- Hallucinations and failure modes
- Robustness and regression testing

📘 `Docs/hallucinations_failure_modes.md`  
📘 `Docs/evaluation_language_models.md`

---

## 🧰 Tools & Frameworks

- Python  
- PyTorch  
- Hugging Face Transformers  
- Jupyter Notebooks  

---

## 📌 Repository Philosophy

- Depth over breadth  
- One research question per notebook  
- Failures are documented explicitly  
- Theory precedes implementation  

> *To understand language models, one must study their structure, training, and limits — not just their outputs.*

---

⭐ *Maintained by Kartik Saroop — AI & Machine Learning Research Learner*
