import streamlit as st
import torch
import os
from unsloth import FastLanguageModel
from transformers import TextStreamer, TextIteratorStreamer
import threading
import base64


# =========================
#   Sidebar Controls
# =========================
st.sidebar.header("Model Settings")
model_name = st.sidebar.text_input("Model Name", value="qwen3_lora_model")
max_new_tokens = st.sidebar.number_input("Max New Tokens", min_value=1, max_value=4096, value=2048)
thinking_mode = st.sidebar.toggle("Enable Thinking Mode", value=True)

# =========================
#   Model Loading
# =========================
@st.cache_resource(show_spinner=True)
def load_model_and_tokenizer(model_name):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer(model_name)

# =========================
#   Chat State
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
#   Main Chat Interface
# =========================
st.title("🧠 Qwen3 Micro-Reasoner")
st.caption(f"Loaded from: {model_name}")


# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a question...")

if prompt:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare messages for model
    messages = st.session_state.messages[-10:]  # last 10 for context
    # Only keep role/content
    chat_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

    # Tokenize with/without thinking
    text = tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking_mode,
    )

    # Stream output
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        # Run generation in a separate thread
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7, top_p=0.8, top_k=20,
            streamer=streamer,
        )
        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # Stream output as it is generated
        for new_text in streamer:
            full_response += new_text
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})