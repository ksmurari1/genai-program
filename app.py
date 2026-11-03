import streamlit as st
from openai import OpenAI

# ---------------------------------------------------------------
# Fine-Tuning Playground  |  Base GPT  vs  Fine-Tuned GPT
# Two independent chats. No shared history. No comparison logic.
# ---------------------------------------------------------------

st.set_page_config(
    page_title="Fine-Tuning Playground",
    page_icon="🧪",
    layout="wide",
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');

    html, body, [class*="css"], .stApp, .stMarkdown, p, label, span, textarea, button {
        font-family: 'Nunito', 'Segoe UI', sans-serif !important;
    }
    .stApp { background-color: #eadef4; }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 2rem; max-width: 1250px; }

    /* Text colour: dark and clearly visible on the lilac background */
    .stApp, .stApp p, .stApp label, .stApp li, .stApp span { color: #2b2140; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #f4ecfa; }

    /* Page title */
    .page-title { font-size: 2rem; font-weight: 800; color: #2b2140; margin-bottom: 0; }
    .page-sub   { font-size: 1rem; color: #5b4d73 !important; margin-bottom: 1.2rem; }

    /* Panel headers */
    .panel-head {
        border-radius: 14px; padding: 14px 18px; text-align: center;
        font-size: 1.35rem; font-weight: 800; margin-bottom: 4px;
    }
    .panel-sub { text-align: center; font-size: 0.92rem; color: #5b4d73 !important; margin-bottom: 10px; }
    .head-base  { background: #dbe9f8; color: #1f4e8c !important; }
    .head-tuned { background: #dff2e6; color: #1f6b3a !important; }

    /* Chat bubbles */
    [data-testid="stChatMessage"] {
        background: #ffffff; border-radius: 14px; padding: 10px 14px;
        border: 1px solid #d9c9e8; margin-bottom: 8px;
    }
    [data-testid="stChatMessage"] p { color: #2b2140 !important; font-size: 1rem; line-height: 1.55; }

    /* Chat window box */
    [data-testid="stVerticalBlockBorderWrapper"] { background: #f7f1fb; border-radius: 14px; }

    /* Input + button */
    textarea {
        background: #ffffff !important; color: #2b2140 !important;
        border: 1px solid #c9b3de !important; border-radius: 12px !important;
    }
    .stButton > button, [data-testid="stFormSubmitButton"] > button {
        background: #6b4a9e; color: #ffffff !important; border: none;
        border-radius: 10px; font-weight: 700; padding: 0.45rem 1.4rem;
    }
    .stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover { background: #563a82; }
    [data-testid="stForm"] { border: none; padding: 0; background: transparent; }

    .footer-note { text-align: center; color: #5b4d73 !important; font-size: 0.9rem; margin-top: 18px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar: settings ----------
try:
    default_key = st.secrets.get("OPENAI_API_KEY", "")
except Exception:
    default_key = ""

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API key", value=default_key, type="password")
    base_model = st.text_input("Base model", value="gpt-4o-mini")
    tuned_model = st.text_input(
        "Fine-tuned model ID",
        placeholder="ft:gpt-4o-mini-2024-07-18:org::abc123",
    )
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    base_system = st.text_area("Base system prompt (optional)", value="", height=80)
    tuned_system = st.text_area("Fine-tuned system prompt (optional)", value="", height=80)
    st.caption("Tip: keep both system prompts the same for a fair test.")

# ---------- Session state: two separate histories ----------
for key in ("base_history", "tuned_history"):
    if key not in st.session_state:
        st.session_state[key] = []


def ask_model(model, history, system_prompt):
    client = OpenAI(api_key=api_key)
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages += history
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature
    )
    return resp.choices[0].message.content


def render_panel(title, subtitle, css_class, history_key, model, system_prompt, form_key):
    st.markdown(f'<div class="panel-head {css_class}">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="panel-sub">{subtitle}</div>', unsafe_allow_html=True)

    chat_box = st.container(height=460, border=True)

    with st.form(form_key, clear_on_submit=True):
        prompt = st.text_area(
            "Your prompt",
            placeholder="Type a prompt...",
            height=90,
            label_visibility="collapsed",
        )
        c1, c2 = st.columns([1, 1])
        send = c1.form_submit_button("Send")
        clear = c2.form_submit_button("Clear chat")

    history = st.session_state[history_key]

    if clear:
        st.session_state[history_key] = []
        st.rerun()

    if send and prompt.strip():
        if not api_key:
            st.warning("Add your OpenAI API key in the sidebar.")
        elif not model.strip():
            st.warning("Enter a model name in the sidebar.")
        else:
            history.append({"role": "user", "content": prompt.strip()})
            try:
                with st.spinner("Thinking..."):
                    reply = ask_model(model.strip(), history, system_prompt)
                history.append({"role": "assistant", "content": reply})
            except Exception as e:
                history.pop()  # remove the unanswered prompt
                st.error(f"API error: {e}")

    with chat_box:
        if not history:
            st.markdown("*No messages yet. Type a prompt below.*")
        for m in history:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])


# ---------- Page ----------
st.markdown('<div class="page-title">Fine-Tuning Playground</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-sub">Chat with the original model and your fine-tuned model side by side.</div>',
    unsafe_allow_html=True,
)

left, right = st.columns(2, gap="large")

with left:
    render_panel(
        "Base GPT", "Original model", "head-base",
        "base_history", base_model, base_system, "form_base",
    )

with right:
    render_panel(
        "Fine-Tuned GPT", "Same base model family", "head-tuned",
        "tuned_history", tuned_model, tuned_system, "form_tuned",
    )

st.markdown(
    '<div class="footer-note">Two independent conversations. No shared history. No response comparison logic.</div>',
    unsafe_allow_html=True,
)
