# 🧠 Qwen3 Micro-Reasoner

> **Lightweight LLM with Thinking Mode • Hybrid Fine-Tuning • Streamlit Chat UI**

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Ashish--kharde1%2FQwen3--Micro--Reasoner-orange)](https://huggingface.co/Ashish-kharde1/Qwen3-Micro-Reasoner)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ashish-kharde1/Micro-Reasoner-Qwen/blob/main/notebooks/qwen3_finetune.ipynb)
[![Powered by Unsloth](https://img.shields.io/badge/⚡%20Unsloth-High%20Efficiency-green)](https://github.com/unslothai/unsloth)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Overview

**Qwen3 Micro-Reasoner** is a 4B parameter model designed to bring "Chain-of-Thought" (CoT) reasoning capabilities to consumer hardware. 

Standard small models often struggle with complex logic. By fine-tuning **Qwen3-4B** on a mix of explicit reasoning data and high-quality instruction sets, this model learns to **"think before it speaks."** It generates an internal monologue wrapped in `<think>` tags to verify its logic before producing a final answer.

---

## 🤖 What Does This Model Do?

This model mimics the behavior of larger reasoning models (like o1 or R1) by splitting generation into two phases:

1.  **The Thinking Phase:**
    The model analyzes the prompt, breaks it down into steps, and checks for errors. This is visible in the internal logs (or via the UI toggle) as:
    ```xml
    <think>
    To solve x^2 + 5x + 6 = 0, I should look for factors of 6 that add up to 5.
    Factors of 6 are 1, 6 and 2, 3.
    2 + 3 = 5. So the factors are (x+2) and (x+3).
    </think>
    ```

2.  **The Response Phase:**
    Based on the thought process, it outputs the clean, final answer:
    > The solutions are x = -2 and x = -3.

---

## 📚 Training Data & Strategy

To balance **deep reasoning** with **general conversational fluency**, this model was trained using a hybrid dataset strategy:

### 1. The Reasoning Core (Logic)
*   **Dataset:** [unsloth/OpenMathReasoning-mini](https://huggingface.co/datasets/unsloth/OpenMathReasoning-mini) (`split="cot"`)
*   **Purpose:** Provides examples containing the `<think>` ... `</think>` structure. This teaches the model *how* to reason step-by-step.

### 2. The Generalist Core (Chat)
*   **Dataset:** [mlabonne/FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k)
*   **Purpose:** High-quality instruction-following data.
*   **Strategy:** A subset of this dataset was mixed in (approx. 25% ratio) to prevent the model from losing its ability to hold normal conversations or handle non-math queries.

---

## 🏗️ Project Features

### ✔️ 1. Efficient Fine-Tuning
*   **Base Model:** `unsloth/Qwen3-4b` (4-bit quantized).
*   **Platform:** Trained entirely on **Google Colab** (T4 GPU).
*   **Optimization:** Uses **Unsloth** for 2x faster training and 60% lower memory usage compared to standard Hugging Face implementations.

### ✔️ 2. Streamlit Chat UI (`app.py`)
*   **Real-time Streaming:** Token-by-token generation.
*   **Thinking Mode Toggle:** A sidebar control to show/hide the raw `<think>` blocks.
*   **Context Memory:** Remembers previous turns in the conversation.

### ✔️ 3. Easy Deployment
*   Fully compatible with `transformers` and `peft`.
*   Can be loaded in 4-bit mode on consumer GPUs (approx. 4GB VRAM required).

---

## ⚙️ Installation & Usage

### 1. Clone & Install
```bash
git clone https://github.com/Ashish-kharde1/Micro-Reasoner-Qwen.git
cd Micro-Reasoner-Qwen

# Install dependencies (Unsloth required for 4-bit inference)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes streamlit
```

### 2. Run the Chat App
```bash
streamlit run app.py
```
---

## 📂 Repository Structure

```text
Micro-Reasoner-Qwen/
├── notebooks/
│   ├── qwen3_finetune.ipynb
    └── fix_widgets.py
├── qwen3_lora_model/
│   ├── added_tokens.json
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── chat_template.jinja
├── .gitattributes
├── .gitignore
├── LICENSE
├── README.md
└── app.py

```

---

## 🧉 Training Details

The full training process is documented in `notebooks/qwen3_finetune.ipynb`.

*   **LoRA Rank (r):** 16
*   **LoRA Alpha:** 32
*   **Target Modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
*   **Learning Rate:** 2e-4
*   **Optimizer:** AdamW 8-bit

---

## 🙌 Acknowledgments

*   **Unsloth AI** for the incredible optimization tools.
*   **Maxime Labonne** for the `FineTome-100k` dataset.
*   **Alibaba Cloud** for the Qwen3 architecture.

---

## 📬 Contact

**Author:** Ashish Kharde  
*   **Hugging Face:** [Ashish-kharde1](https://huggingface.co/Ashish-kharde1)  
*   **GitHub:** [Ashish-kharde1](https://github.com/Ashish-kharde1)