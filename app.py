import streamlit as st
import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer, TextIteratorStreamer
import threading
import base64

# =========================
#   Sidebar Controls
# =========================
st.sidebar.header("Model Settings")

# CHANGE THIS to your Hugging Face Username/ModelRepo
default_model = "Ashish-kharde1/Qwen3-Micro-Reasoner" 

model_name = st.sidebar.text_input("Model Name (Hugging Face Path)", value=default_model)
max_new_tokens = st.sidebar.number_input("Max New Tokens", min_value=1, max_value=4096, value=1024)
thinking_mode = st.sidebar.toggle("Enable Thinking Mode (<think>)", value=True)

# =========================
#   Model Loading
# =========================
@st.cache_resource(show_spinner="Downloading Model from Hugging Face...")
def load_model_and_tokenizer(name):
    # This automatically downloads Base Model + Adapters
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = name,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    return model, tokenizer

try:
    model, tokenizer = load_model_and_tokenizer(model_name)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# =========================
#   Chat Interface
# =========================
st.title("🧠 Qwen3 Micro-Reasoner")
st.caption(f"Loaded from: {model_name}")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I am ready. Ask me a math problem!"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a question...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Tokenize with thinking parameter
    # We apply the template but keep it as tensors on GPU
    inputs = tokenizer.apply_chat_template(
        st.session_state.messages,
        tokenize = True,
        add_generation_prompt = True,
        return_tensors = "pt",
        enable_thinking = thinking_mode,
    ).to("cuda")

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Streamer setup
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, decode_kwargs={"errors": "ignore"})
        
        # Generation arguments
        generation_kwargs = dict(
            input_ids = inputs,
            streamer = streamer,
            max_new_tokens = max_new_tokens,
            use_cache = True,
            temperature = 0.8,
        )
        
        # Threading for streaming
        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # Render stream
        for new_text in streamer:
            full_response += new_text
            message_placeholder.markdown(full_response + "▌")
            
        message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})