from utils.settings import (
    AZURE_OPENAI_MODEL_NAME,
    RETRIEVAL_CONFIDENCE_THRESHOLD,
    RETRIEVAL_SKIP_VALIDATION_THRESHOLD,
    VALIDATION_CONFIDENCE_THRESHOLD,
)
import streamlit as st
import requests
import os
from pathlib import Path

from app.generation.query_utils import select_relevant_image_path
from utils.settings import RETRIEVAL_CONFIDENCE_THRESHOLD, VALIDATION_CONFIDENCE_THRESHOLD
FASTAPI_URL = os.getenv("FASTAPI_URL1", "http://127.0.0.1:8004")

# Auto-probe active FastAPI port (fallback to port 8000 if 8004 is unreachable)
try:
    resp = requests.get(f"{FASTAPI_URL}/health", timeout=0.5)
    if resp.status_code != 200:
        raise Exception()
except Exception:
    try:
        fallback_url = "http://127.0.0.1:8006"
        resp = requests.get(f"{fallback_url}/health", timeout=0.5)
        if resp.status_code == 200:
            FASTAPI_URL = fallback_url
    except Exception:
        pass

# Page Config
st.set_page_config(
    page_title="InSightDocs — Interactive RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling Injection
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Hide image enlarge / fullscreen icon overlay */
    button[title="View fullscreen"],
    button[title="Enlarge image"],
    [data-testid="stElementToolbar"],
    .stApp button[aria-label="View fullscreen"],
    .stApp button[title="View fullscreen"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }
    
    /* Display crisp intermediate-sized image without pixel stretching */
    [data-testid="stImage"] img {
        max-width: 100% !important;
        width: auto !important;
        height: auto !important;
        max-height: 520px !important;
        object-fit: contain !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25) !important;
        margin-top: 0.5rem !important;
    }
    
    /* Headers */
    .title-container {
        padding: 1.5rem 0rem;
        text-align: left;
    }
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(to right, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.05em;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Messages styling */
    .chat-container {
        max-width: 850px;
        margin: 0 auto;
        padding: 2rem 0;
    }
    .chat-bubble {
        padding: 1.25rem 1.75rem;
        border-radius: 16px;
        margin-bottom: 1.25rem;
        line-height: 1.6;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.05);
        font-size: 1.05rem;
    }
    .user-bubble {
        background: rgba(99, 102, 241, 0.15);
        color: #e2e8f0;
        margin-left: 20%;
        border-bottom-right-radius: 4px;
        border-right: 3px solid #6366f1;
    }
    .assistant-bubble {
        background: rgba(30, 41, 59, 0.7);
        color: #f1f5f9;
        margin-right: 20%;
        border-bottom-left-radius: 4px;
        border-left: 3px solid #c084fc;
        backdrop-filter: blur(12px);
    }
    
    /* Input area */
    div[data-testid="stForm"] {
        background-color: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
    }
    
    /* Decent Grey-Purple with White Text Styling */
    /* Standard button styling to prevent white-on-white and maintain theme */
    button {
        background-color: #25213b !important;
        color: #f8fafc !important;
        border: 1px solid rgba(139, 92, 246, 0.4) !important;
        border-radius: 8px !important;
        transition: background 0.3s ease, border-color 0.3s ease !important;
    }
    button:hover, button:active, button:focus {
        background-color: #312e81 !important;
        color: #ffffff !important;
        border-color: #8b5cf6 !important;
    }

    [data-testid="stFileUploader"] section {
        background-color: #25213b !important;
        color: #f8fafc !important;
        border: 1px dashed rgba(139, 92, 246, 0.4) !important;
        border-radius: 8px !important;
    }
    [data-testid="stFileUploader"] label {
        color: #e2e8f0 !important;
    }
    [data-testid="stFileUploader"] section p, 
    [data-testid="stFileUploader"] section span, 
    [data-testid="stFileUploader"] section small {
        color: #cbd5e1 !important;
    }
    [data-testid="stFileUploader"] button {
        background-color: #6d28d9 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.25rem 1rem !important;
        transition: background 0.3s ease !important;
    }
    [data-testid="stFileUploader"] button:hover {
        background-color: #7c3aed !important;
    }
    
    .stTextInput input {
        background-color: #25213b !important;
        color: #f8fafc !important;
        border: 1px solid rgba(139, 92, 246, 0.4) !important;
        border-radius: 8px !important;
    }
    .stTextInput input::placeholder {
        color: #94a3b8 !important;
    }
    .stTextInput label {
        color: #e2e8f0 !important;
    }

    /* Selection highlight style */
    ::selection {
        background-color: #6d28d9 !important;
        color: #ffffff !important;
    }
    ::-moz-selection {
        background-color: #6d28d9 !important;
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State for Chat History and Session ID
import uuid
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

# Sidebar Content
with st.sidebar:
    st.markdown("<h2 style='color: #818cf8; font-weight: 800; margin-bottom: 1rem;'>📁 Document Ingestion</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload research PDF to the RAG pipeline:", type=["pdf"])
    
    col1, col2 = st.columns(2)
    with col1:
        force_reprocess = st.checkbox("Force Ingest", value=False)
    
    if uploaded_file is not None:
        if st.button("🚀 Process & Ingest", use_container_width=True):
            with st.spinner("Extracting, describing, and embedding doc..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    data = {"force": str(force_reprocess).lower()}
                    
                    response = requests.post(f"{FASTAPI_URL}/ingest", files=files, data=data)
                    
                    if response.status_code == 200:
                        res_json = response.json()
                        st.success(f"Successfully ingested '{uploaded_file.name}'!")
                        st.json(res_json)
                    else:
                        st.error(f"Ingestion failed (Code {response.status_code}): {response.text}")
                except Exception as e:
                    st.error(f"Error connecting to backend: {e}")
  
    try:
        doc_resp = requests.get(f"{FASTAPI_URL}/documents", timeout=3)
        if doc_resp.status_code == 200:
            doc_data = doc_resp.json()
            docs = doc_data.get("documents", {})
            st.caption(f"Total Chunks Ingested: **{doc_data.get('total_chunks', 0)}**")
            with st.expander(f"Uploaded Papers ({len(docs)})"):
                for doc_name, count in docs.items():
                    st.write(f"- **{doc_name[:35]}...** ({count} chunks)")
        else:
            st.caption("Could not list corpus documents.")
    except Exception:
        st.caption("Corpus document retrieval unavailable.")

    st.markdown("---")
    st.markdown("<h3 style='color: #c084fc; font-weight: 800; margin-bottom: 0.5rem;'>⚙️ Backend Health</h3>", unsafe_allow_html=True)
    try:
        health_resp = requests.get(f"{FASTAPI_URL}/health", timeout=3)
        if health_resp.status_code == 200:
            st.markdown("🟢 **FastAPI Server**: Running")
        else:
            st.markdown("🔴 **FastAPI Server**: Unreachable")
    except Exception:
        st.markdown("🔴 **FastAPI Server**: Offline")

    st.markdown("---")
    if st.session_state.messages:
        import json as _json
        chat_export = _json.dumps(st.session_state.messages, indent=2)
        st.download_button(
            label="📥 Export Chat Log (JSON)",
            data=chat_export,
            file_name=f"InSightDocs_chat_session_{st.session_state.session_id}.json",
            mime="application/json",
            use_container_width=True,
        )

# Main Interface
st.markdown(
    """
    <div class='title-container'>
        <h1 class='main-title'>⚡ InSightDocs Assistant</h1>
        <div class='sub-title'>Advanced Multimodal RAG Pipeline for Scientific & Technical Document Analysis</div>
    </div>
    """,
    unsafe_allow_html=True,
)

import re

def _to_html(text: str) -> str:
    """Convert markdown text to clean HTML for rendering inside custom styled chat bubbles."""
    if not text:
        return ""
    # Extract clean answer if text is wrapped in JSON
    if isinstance(text, str) and text.strip().startswith("{") and '"answer"' in text:
        try:
            import json
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "answer" in parsed:
                text = parsed["answer"]
        except Exception:
            pass

    # Strip any stray div tags from LLM or markdown artifacts
    text = re.sub(r"</?div.*?>", "", str(text), flags=re.IGNORECASE).strip()

    def _normalize_latex(s: str) -> str:
        s = s.replace('\\$', '$')
        repl = {
            '\\rightarrow': '→',
            '\\to': '→',
            '\\gamma': 'γ',
            '\\times': '×',
            '\\leq': '≤',
            '\\geq': '≥',
            '\\top': '⊤',
            '\\alpha': 'α',
            '\\beta': 'β',
            '\\cdot': '·',
            '\\sqrt': '√',
            '\\sum': '∑',
            '\\hat': '^',
        }
        for k, v in repl.items():
            s = s.replace(k, v)
        # Clean stray backslashes before plain letters
        s = re.sub(r'\\([a-zA-Z])', r'\1', s)
        return s

    try:
        import markdown
        html = markdown.markdown(text, extensions=["tables", "fenced_code"])
        html = re.sub(r"</?div.*?>", "", html, flags=re.IGNORECASE).strip()
        return _normalize_latex(html)
    except Exception:
        formatted = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
        formatted = formatted.replace('\n', '<br/>')
        formatted = re.sub(r"</?div.*?>", "", formatted, flags=re.IGNORECASE).strip()
        return _normalize_latex(formatted)



for i, msg in enumerate(st.session_state.messages):
    role_class = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
    sender_name = "You" if msg["role"] == "user" else "Assistant"
    content_html = _to_html(msg["content"])

    user_img_html = ""
    if msg["role"] == "user":
        img_b64 = msg.get("user_image_b64")
        if not img_b64 and msg.get("user_image_bytes"):
            import base64
            img_b64 = f"data:image/png;base64,{base64.b64encode(msg['user_image_bytes']).decode('utf-8')}"
        if img_b64:
            user_img_html = f'<br/><br/><img src="{img_b64}" style="max-width: 100%; max-height: 280px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.15); display: block;" />'

    st.markdown(
        f"""
        <div class="chat-bubble {role_class}">
            <b style="font-size: 1.05rem; letter-spacing: 0.02em;">{sender_name}:</b><br/><br/>
            {content_html}
            {user_img_html}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Display retrieved/generated image for the assistant message (persist across history)
    if msg["role"] == "assistant" and msg.get("retrieved_image_path"):
        img_p = Path(msg["retrieved_image_path"])
        if img_p.exists():
            st.image(str(img_p), caption="Retrieved / Generated Source Image", width=580)


# Chat Input Form
with st.form("chat_form", clear_on_submit=True):
    user_query = st.text_input("Ask a question about the uploaded documents:", placeholder="e.g., What is the main contribution of VANDERER?")
    uploaded_image = st.file_uploader("Attach a PDF or image file to query about (optional):", type=["pdf", "png", "jpg", "jpeg"])
    submit_button = st.form_submit_button("Send Query")

    if submit_button and (user_query.strip() or uploaded_image):
        import base64
        image_base64 = None
        user_content = user_query.strip() or "[Uploaded Image Query]"

        user_image_b64 = None
        if uploaded_image:
            img_bytes = uploaded_image.getvalue()
            b64_str = base64.b64encode(img_bytes).decode("utf-8")
            mime_type = uploaded_image.type or "image/png"
            user_image_b64 = f"data:{mime_type};base64,{b64_str}"

        # Append User Message with image bytes and b64 data URL
        st.session_state.messages.append({
            "role": "user",
            "content": user_content,
            "has_image": bool(uploaded_image),
            "user_image_bytes": uploaded_image.getvalue() if uploaded_image else None,
            "user_image_b64": user_image_b64,
        })

        # Call Backend
        with st.spinner("thinking..."):
            try:
                data = {
                    "query": user_query.strip(),
                    "session_id": st.session_state.session_id,
                }
                files = {}
                if uploaded_image:
                    files["image"] = (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)
                res = requests.post(
                    f"{FASTAPI_URL}/query",
                    data=data,
                    files=files if files else None,
                )
                if res.status_code == 200:
                    res_data = res.json()
                    answer = res_data.get("answer", "No answer received.")
                    if isinstance(answer, str) and answer.strip().startswith("{") and '"answer"' in answer:
                        try:
                            import json as _json_module, re as _re_module
                            _match = _re_module.search(r"\{.*\}", answer, _re_module.S)
                            _parsed = _json_module.loads(_match.group(0)) if _match else _json_module.loads(answer)
                            if isinstance(_parsed, dict) and "answer" in _parsed:
                                answer = _parsed["answer"]
                        except Exception:
                            pass
                    chunks = res_data.get("chunks") or []

                    retrieved_image_path = res_data.get("retrieved_image_path")
                    if (
                        not retrieved_image_path
                        and chunks
                        and res_data.get("route") == "rag"
                        and user_query.strip()
                        and not uploaded_image
                    ):
                        retrieved_image_path = select_relevant_image_path(
                            user_query.strip(),
                            chunks,
                        )
                    if retrieved_image_path and not Path(retrieved_image_path).exists():
                        retrieved_image_path = None

                    # Stream answer word-by-word inside the custom assistant bubble
                    message_placeholder = st.empty()
                    words = answer.split(" ")
                    full_response = ""
                    for word in words:
                        full_response += word + " "
                        message_placeholder.markdown(
                            f"""
                            <div class="chat-container">
                                <div class="chat-bubble assistant-bubble">
                                    <b style="font-size: 1.05rem; letter-spacing: 0.02em;">Assistant:</b><br/><br/>
                                    {_to_html(full_response)}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        import time
                        time.sleep(0.02)

                    # Final render — ensure complete HTML markdown rendering
                    final_html = _to_html(answer)
                    message_placeholder.markdown(
                        f"""
                        <div class="chat-container">
                            <div class="chat-bubble assistant-bubble">
                                <b style="font-size: 1.05rem; letter-spacing: 0.02em;">Assistant:</b><br/><br/>
                                {final_html}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Save assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "retrieved_image_path": retrieved_image_path,
                    })
                    st.rerun()
                else:
                    st.error(f"Error querying backend (Code {res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Error communicating with FastAPI server: {e}")



