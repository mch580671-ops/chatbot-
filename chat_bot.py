# streamlit_chat_no_chain.py
import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from typing import Any

load_dotenv()

# -------------------------
# Config
# -------------------------
st.set_page_config(page_title="MindFlow Nova", layout="wide", page_icon="🤖")
st.title("🤖 MindFlow Nova")

# -------------------------
# Watermark CSS 
# -------------------------
watermark_style = """
<style>
.watermark {
    position: fixed;
    bottom: 10px;
    right: 15px;
    color: rgba(255, 255, 255, 0.55);
    background: rgba(0, 0, 0, 0.35);
    padding: 6px 12px;
    border-radius: 8px;
    font-size: 14px;
    z-index: 9999;
    pointer-events: none;
    user-select: none;
}
</style>

<div class="watermark">Developed by MADHU</div>
"""
st.markdown(watermark_style, unsafe_allow_html=True)

# Get API key
def get_api_key():
    return os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

# -------------------------
# Token Redaction Helper
# -------------------------
def redact_tokens(obj: Any):
    if isinstance(obj, dict):
        redacted = {}
        for k, v in obj.items():
            kl = k.lower()
            if any(substr in kl for substr in ("token", "usage", "logprob", "logprobs", "raw", "hidden")):
                redacted[k] = "<REDACTED>"
            else:
                redacted[k] = redact_tokens(v)
        return redacted
    elif isinstance(obj, list):
        return [redact_tokens(x) for x in obj]
    else:
        return obj

# -------------------------
# Extract assistant text safely
# -------------------------
def extract_assistant_text(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result

    if isinstance(result, dict):
        for key in ("response", "output", "text", "content"):
            if key in result and isinstance(result[key], (str, int, float)):
                return str(result[key])

        gens = result.get("generations") or result.get("generation")
        if gens:
            try:
                cand = gens[0][0] if isinstance(gens[0], list) else gens[0]
                if isinstance(cand, dict):
                    return cand.get("text") or cand.get("content") or ""
                return getattr(cand, "text", None) or getattr(cand, "content", None) or ""
            except:
                pass

    for attr in ("response", "output", "text"):
        val = getattr(result, attr, None)
        if isinstance(val, (str, int, float)):
            return str(val)

    try:
        s = str(result)
        low = s.lower()
        if any(t in low for t in ("usage", "token", "logprob", "hidden")):
            return "[Model returned a complex object — textual output extracted.]"
        return s
    except:
        return ""

# -------------------------
# Initialize LLM
# -------------------------
if "llm" not in st.session_state:
    groq_api_key = get_api_key()
    if not groq_api_key:
        st.error("GROQ_API_KEY not found. Add it to .env or Streamlit secrets.")
        st.stop()
    try:
        st.session_state["llm"] = ChatGroq(
            api_key=groq_api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=None
        )
    except Exception as e:
        st.exception(f"Failed to initialize ChatGroq: {e}")
        st.stop()

# Chat history
if "history" not in st.session_state:
    st.session_state["history"] = []
    st.session_state["system_prompt"] = "You are a helpful, concise assistant."

# Sidebar
with st.sidebar:
    st.header("Settings")
    temp = st.slider("Temperature", 0.0, 1.0, 0.7)
    model = st.text_input("Model name", value="llama-3.3-70b-versatile")
    st.checkbox("Show raw LLM response", key="show_raw_response", value=False)
    st.markdown("---")
    st.write("Tip: Put GROQ_API_KEY in .env")

# Apply updated settings
try:
    st.session_state["llm"].temperature = temp
    st.session_state["llm"].model = model
except:
    pass

# -------------------------
# Chat UI
# -------------------------
chat_col, ctrl_col = st.columns([4, 1])

with chat_col:
    st.subheader("Chat")

    for msg in st.session_state["history"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Type your message...")

with ctrl_col:
    if st.button("Clear chat"):
        st.session_state["history"] = []
        st.rerun()

# When user sends a message
if user_input:
    st.session_state["history"].append({"role": "user", "content": user_input})
    st.rerun()

# -------------------------
# Generate assistant response
# -------------------------
if st.session_state["history"] and st.session_state["history"][-1]["role"] == "user":
    llm = st.session_state["llm"]

    messages = [{"role": "system", "content": st.session_state["system_prompt"]}]
    messages.extend(st.session_state["history"])

    placeholder = st.empty()
    with placeholder.container():
        st.chat_message("assistant").write("...")

    try:
        result = llm.invoke(messages)
    except Exception as e:
        st.session_state["history"].append({"role": "assistant", "content": f"Error: {e}"})
        placeholder.empty()
        st.rerun()

    assistant_text = extract_assistant_text(result)
    if not assistant_text:
        assistant_text = "Sorry — I couldn't parse the response."

    st.session_state["history"].append({"role": "assistant", "content": assistant_text})

    placeholder.empty()

    if st.session_state.get("show_raw_response"):
        st.write("Raw model response (redacted):")
        st.json(redact_tokens(result))

    st.rerun()