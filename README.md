# 🚀 Micro-Reasoner-Qwen

> **Lightweight reasoning-capable LLM built on Qwen3-4B using efficient LoRA fine-tuning**

**Micro-Reasoner-Qwen** is a compact reasoning-enhanced language model based on **Qwen3-4B**. It is fine-tuned to explicitly generate *chain-of-thought (CoT) reasoning* before producing a final answer, enabling improved multi-step reasoning while remaining lightweight and fast.

This project focuses on **practical reasoning**, **low-resource inference**, and **developer-friendly experimentation**.

---

## 🖼️ Demo

The screenshot below shows the Streamlit chat UI with reasoning (`<think>`) and the final answer output.

![Chat UI Screenshot](/assets/ui-demo.png)

---

## ⚡ Quick Start (5 Minutes)

```bash
# Clone the repository
git clone https://github.com/Ashish-kharde1/Micro-Reasoner-Qwen.git
cd Micro-Reasoner-Qwen

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

---

## 📌 Key Features

*   **🧠 Explicit Reasoning (CoT):** The model generates reasoning wrapped in `<think>...</think>` tags before providing the final answer.
*   **⚡ Efficient 4-bit Inference:** Runs comfortably on consumer GPUs using Unsloth + bitsandbytes.
*   **🔧 LoRA Fine-Tuning:** Parameter-efficient training with minimal memory overhead.
*   **💬 Interactive Chat UI:** A Streamlit-based interface featuring conversation memory and token streaming.

---

## 🧠 How It Works

### Base Model
Built on **Qwen3-4B** for strong instruction-following capabilities and multilingual support.

### Hybrid Training
Fine-tuned using a specific mix of data:
1.  **Reasoning datasets:** Math, logic puzzles, and step-by-step explanations.
2.  **General instruction-following data:** To maintain conversational fluency.

### Reasoning Output
The model explicitly emits reasoning tokens before the final response.

> **⚠️ Important:** The `<think>` reasoning is explicit generated text, not hidden internal latent states.

---

## 💻 Hardware Requirements

| Component | Requirement |
| :--- | :--- |
| **GPU** | Minimum **6 GB VRAM** (4-bit quantization) |
| **Recommended** | 8–12 GB VRAM |
| **CPU-only** | ❌ Not supported |
| **RAM** | ≥ 16 GB |

---

## ▶️ Running the Chat Application

To start the interactive interface:

```bash
streamlit run app.py
```

**UI Features:**
*   Token-streaming responses for real-time feedback.
*   Context-aware conversation history.
*   Toggleable display for the reasoning process.

---

## 📁 Repository Structure

```text
Micro-Reasoner-Qwen/
├── notebooks/
│   ├── qwen3_finetune.ipynb     # Training notebook
│   └── fix_widgets.py           # Colab widget fix
├── qwen3_lora_model/
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   └── chat_template.jinja
├── app.py                       # Streamlit chat app
├── requirements.txt             # Python dependencies
├── LICENSE
└── README.md
```

---

## 🧪 Training Details

The training process is documented in `notebooks/qwen3_finetune.ipynb`.

**LoRA Configuration (Typical):**

| Parameter | Value |
| :--- | :--- |
| **Base Model** | unsloth/Qwen3-4b |
| **LoRA Rank** | 16 |
| **LoRA Alpha** | 32 |
| **Target Modules** | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| **Optimizer** | AdamW (8-bit) |
| **Learning Rate** | 2e-4 |

*This setup is designed to balance deep reasoning ability with conversational fluency.*

---

## ⚠️ Limitations

*   Not a replacement for large-scale reasoning models (e.g., o1, DeepSeek-R1).
*   Reasoning quality is dependent on prompt structure.
*   No formal benchmark results included yet.
*   Long-context reasoning may degrade beyond ~4k tokens.

---

## 📦 Dependencies

Key libraries used in this project:
*   `unsloth`
*   `peft`
*   `accelerate`
*   `bitsandbytes`
*   `streamlit`

*Refer to `requirements.txt` for exact version numbers.*

---

## 🧠 Credits & Acknowledgements

*   **Unsloth AI** — For efficient 4-bit training & inference tools.
*   **Qwen Team (Alibaba Cloud)** — For the powerful base model architecture.
*   **Maxime Labonne** — For the FineTome-100k reasoning dataset.

---

## 📝 License

Distributed under the **MIT License**.

---

## 📬 Contact

**Author:** Ashish Kharde

*   **GitHub:** [https://github.com/Ashish-kharde1](https://github.com/Ashish-kharde1)
*   **Hugging Face:** [https://huggingface.co/Ashish-kharde1](https://huggingface.co/Ashish-kharde1)

---

## ⭐ Contributing

Contributions, issues, and PRs are welcome! Feel free to propose improvements, benchmarks, or UI enhancements.
