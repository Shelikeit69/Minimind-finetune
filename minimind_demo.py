"""
MiniMind 财务领域微调 对比 Demo
用法：streamlit run minimind_demo.py --server.port 6006 --server.address 0.0.0.0
"""

import time
import warnings
import torch
import streamlit as st
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

warnings.filterwarnings("ignore")

# ── 页面配置 ────────────────────────────────────────────────
st.set_page_config(
    page_title="MiniMind · Finance Fine-tuning Demo",
    page_icon="🧠",
    layout="wide",
)

# ── Windows 经典样式 ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #d4d0c8;
    color: #000000;
    font-family: 'Noto Sans SC', 'Microsoft Sans Serif', Tahoma, sans-serif;
    font-size: 13px;
}
[data-testid="stSidebar"] { display: none; }
#MainMenu, footer, header { visibility: hidden; }

/* 窗口外框 */
.win-window {
    background: #d4d0c8;
    border: 2px solid #ffffff;
    border-right-color: #404040;
    border-bottom-color: #404040;
    box-shadow: inset 1px 1px 0 #ffffff, inset -1px -1px 0 #808080;
    margin: 16px auto;
    max-width: 1200px;
    padding: 0;
}

/* 标题栏 */
.win-titlebar {
    background: linear-gradient(to right, #000080, #1084d0);
    color: white;
    padding: 4px 6px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-weight: bold;
    font-size: 13px;
    font-family: 'Noto Sans SC', Tahoma, sans-serif;
    user-select: none;
}
.win-titlebar-left {
    display: flex;
    align-items: center;
    gap: 6px;
}
.win-titlebar-btns {
    display: flex;
    gap: 2px;
}
.win-btn {
    background: #d4d0c8;
    border: 1px solid #ffffff;
    border-right-color: #404040;
    border-bottom-color: #404040;
    color: #000;
    width: 18px;
    height: 16px;
    font-size: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: default;
    line-height: 1;
}

/* 窗口内容区 */
.win-body {
    background: #d4d0c8;
    padding: 10px 12px 12px;
}

/* 信息栏 */
.win-infobar {
    background: #d4d0c8;
    border: 1px solid #808080;
    border-top-color: #404040;
    border-left-color: #404040;
    padding: 6px 10px;
    margin-bottom: 10px;
    font-size: 12px;
    color: #000;
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
}
.win-infobar-label {
    font-weight: bold;
    color: #000080;
}
.win-chip {
    background: #ffffff;
    border: 1px solid #808080;
    border-top-color: #404040;
    border-left-color: #404040;
    padding: 1px 8px;
    font-size: 11px;
    font-family: 'Courier New', monospace;
}

/* 回答面板 */
.answer-panel {
    border: 2px solid #ffffff;
    border-right-color: #404040;
    border-bottom-color: #404040;
    box-shadow: inset 1px 1px 0 #808080;
    margin-bottom: 4px;
}
.answer-titlebar-base {
    background: #808080;
    color: white;
    padding: 3px 8px;
    font-size: 12px;
    font-weight: bold;
    font-family: 'Noto Sans SC', Tahoma, sans-serif;
}
.answer-titlebar-finance {
    background: linear-gradient(to right, #000080, #1084d0);
    color: white;
    padding: 3px 8px;
    font-size: 12px;
    font-weight: bold;
    font-family: 'Noto Sans SC', Tahoma, sans-serif;
}
.answer-body {
    background: #ffffff;
    border: 1px solid #808080;
    border-top: none;
    padding: 10px 12px;
    min-height: 200px;
    font-size: 13px;
    line-height: 1.8;
    color: #000000;
    white-space: pre-wrap;
    font-family: 'Noto Sans SC', Tahoma, sans-serif;
}
.answer-placeholder {
    color: #808080;
    font-style: italic;
}

/* 状态栏 */
.win-statusbar {
    background: #d4d0c8;
    border-top: 1px solid #808080;
    padding: 3px 8px;
    font-size: 11px;
    color: #000;
    margin-top: 12px;
}
.win-status-panel {
    border: 1px solid #808080;
    border-top-color: #404040;
    border-left-color: #404040;
    padding: 2px 8px;
    font-size: 11px;
}

/* Streamlit 按钮 */
div[data-testid="stButton"] > button {
    background: #d4d0c8 !important;
    border: 2px solid !important;
    border-color: #ffffff #404040 #404040 #ffffff !important;
    box-shadow: inset 1px 1px 0 #dfdfdf, inset -1px -1px 0 #808080 !important;
    color: #000000 !important;
    border-radius: 0 !important;
    font-family: 'Noto Sans SC', Tahoma, sans-serif !important;
    font-size: 11px !important;
    padding: 3px 6px !important;
    min-height: 24px !important;
    height: auto !important;
    transition: none !important;
    white-space: normal !important;
    line-height: 1.4 !important;
    text-align: center !important;
}
div[data-testid="stButton"] > button:hover {
    background: #d4d0c8 !important;
}
div[data-testid="stButton"] > button:active {
    border-color: #404040 #ffffff #ffffff #404040 !important;
    box-shadow: inset -1px -1px 0 #dfdfdf, inset 1px 1px 0 #808080 !important;
}

/* 输入框 */
div[data-testid="stTextInput"] input {
    background: #ffffff !important;
    border: 2px solid !important;
    border-color: #404040 #ffffff #ffffff #404040 !important;
    box-shadow: inset 1px 1px 2px #808080 !important;
    border-radius: 0 !important;
    color: #000 !important;
    font-family: 'Noto Sans SC', Tahoma, sans-serif !important;
    font-size: 13px !important;
    padding: 4px 6px !important;
}
div[data-testid="stTextInput"] input:focus {
    outline: none !important;
    box-shadow: inset 1px 1px 2px #808080 !important;
}

div[data-testid="stColumns"] { gap: 8px; }
</style>
""", unsafe_allow_html=True)

# ── 常量 ────────────────────────────────────────────────────
HIDDEN_SIZE = 768
NUM_LAYERS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "out"

PRESET_QUESTIONS = [
    ("ROE 是什么？怎么解读？", "What is ROE & how to read it?"),
    ("Free Cash Flow 和净利润有什么区别？", "FCF vs. Net Income — difference?"),
    ("DCF 估值是什么原理？", "How does DCF valuation work?"),
    ("EV/EBITDA 和 P/E 哪个更好用？", "EV/EBITDA vs. P/E — which is better?"),
    ("如何用财报判断公司是否在做盈余管理？", "How to detect earnings management?"),
    ("净利润为正但公司却破产，这可能发生吗？", "Can a profitable company go bankrupt?"),
    ("什么是 WACC？", "What is WACC?"),
    ("如何快速判断一份财报的质量好不好？", "How to quickly assess report quality?"),
]

# ── 模型加载 ─────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("model")

    def _load(weight_name):
        cfg = MiniMindConfig(hidden_size=HIDDEN_SIZE, num_hidden_layers=NUM_LAYERS)
        m = MiniMindForCausalLM(cfg)
        ckp = f"./{SAVE_DIR}/{weight_name}_{HIDDEN_SIZE}.pth"
        m.load_state_dict(torch.load(ckp, map_location=DEVICE), strict=True)
        return m.half().eval().to(DEVICE)

    return tokenizer, _load("full_sft"), _load("finance_sft"), _load("sentiment_sft")

# ── 推理函数 ─────────────────────────────────────────────────
def generate(model, tokenizer, prompt: str):
    torch.manual_seed(42)
    conversation = [{"role": "user", "content": prompt}]
    inputs_text = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(inputs_text, return_tensors="pt", truncation=True).to(DEVICE)
    t0 = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=512,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_p=0.95,
            temperature=0.85,
            repetition_penalty=1.0,
        )
    elapsed = time.time() - t0
    response = tokenizer.decode(
        generated_ids[0][len(inputs["input_ids"][0]):],
        skip_special_tokens=True
    )
    speed = (len(generated_ids[0]) - len(inputs["input_ids"][0])) / elapsed if elapsed > 0 else 0
    return response, speed

# ── Session State 初始化 ─────────────────────────────────────
for key, default in [
    ("question", ""),
    ("base_answer", ""),
    ("finance_answer", ""),
    ("base_speed", 0.0),
    ("finance_speed", 0.0),
    ("sentiment_input", ""),
    ("sentiment_result", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── 加载模型 ─────────────────────────────────────────────────
with st.spinner("Loading models / 正在加载模型…"):
    tokenizer, model_base, model_finance, model_sentiment = load_models()

# ── 标题窗口 ─────────────────────────────────────────────────
st.markdown("""
<div class="win-window">
  <div class="win-titlebar">
    <div class="win-titlebar-left">
      🧠&nbsp; MiniMind · Finance Fine-tuning Demo &nbsp;／&nbsp; 财务领域微调对比
    </div>
    <div class="win-titlebar-btns">
      <div class="win-btn">_</div>
      <div class="win-btn">□</div>
      <div class="win-btn">✕</div>
    </div>
  </div>
  <div class="win-body">
    <div class="win-infobar">
      <span><span class="win-infobar-label">Model：</span>MiniMind</span>
      <span class="win-chip">hidden_size = 768</span>
      <span class="win-chip">num_layers = 8</span>
      <span class="win-chip">params = 63.9M</span>
      <span class="win-chip">finance data = 23 条</span>
      <span class="win-chip">fine-tune epochs = 10</span>
      <span class="win-chip">total cost = ¥50</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── 预设问题 ─────────────────────────────────────────────────
st.markdown("**预设问题 · Preset Questions** — 点击填入 / Click to fill")
cols = st.columns(4)
for i, (zh, en) in enumerate(PRESET_QUESTIONS):
    if cols[i % 4].button(f"{zh}\n{en}", key=f"preset_{i}", use_container_width=True):
        st.session_state.question = zh
        st.rerun()

# ── 输入区 ───────────────────────────────────────────────────
st.markdown("**输入问题 · Enter Question**")
question_val = st.text_input(
    label="question",
    value=st.session_state.question,
    placeholder="请输入财务问题 / Enter a finance question…",
    label_visibility="collapsed",
)
st.session_state.question = question_val

run_col, _ = st.columns([2, 5])
run_clicked = run_col.button("▶  对比推理 · Run Comparison", use_container_width=True)

# ── 推理 ─────────────────────────────────────────────────────
if run_clicked and st.session_state.question.strip():
    q = st.session_state.question.strip()
    with st.spinner("full_sft 推理中… / Generating with full_sft…"):
        ans_base, spd_base = generate(model_base, tokenizer, q)
    st.session_state.base_answer = ans_base
    st.session_state.base_speed = spd_base
    with st.spinner("finance_sft 推理中… / Generating with finance_sft…"):
        ans_fin, spd_fin = generate(model_finance, tokenizer, q)
    st.session_state.finance_answer = ans_fin
    st.session_state.finance_speed = spd_fin

# ── 结果展示 ─────────────────────────────────────────────────
st.markdown("---")
col_base, col_fin = st.columns(2)

with col_base:
    body = st.session_state.base_answer or '<span class="answer-placeholder">等待提问… / Waiting for input…</span>'
    st.markdown(f"""
    <div class="answer-panel">
      <div class="answer-titlebar-base">⬜ FULL_SFT · 通用模型 / General Model</div>
      <div class="answer-body">{body}</div>
    </div>
    """, unsafe_allow_html=True)
    if st.session_state.base_speed > 0:
        st.caption(f"⚡ {st.session_state.base_speed:.1f} tokens/s")

with col_fin:
    body = st.session_state.finance_answer or '<span class="answer-placeholder">等待提问… / Waiting for input…</span>'
    st.markdown(f"""
    <div class="answer-panel">
      <div class="answer-titlebar-finance">🟡 FINANCE_SFT · 财务微调 / Finance Fine-tuned</div>
      <div class="answer-body">{body}</div>
    </div>
    """, unsafe_allow_html=True)
    if st.session_state.finance_speed > 0:
        st.caption(f"⚡ {st.session_state.finance_speed:.1f} tokens/s")

# ── 情感分析区块 ─────────────────────────────────────────────
st.markdown("---")
st.markdown("**情感分析 · Sentiment Analysis** — 输入英文金融新闻句子 / Enter an English financial news sentence")

sent_input = st.text_input(
    label="sentiment_input_label",
    label_visibility="collapsed",
    value=st.session_state.sentiment_input,
    placeholder="e.g. The company reported record profits this quarter.",
    key="sentiment_text"
)
st.session_state.sentiment_input = sent_input

sent_col, _ = st.columns([2, 5])
with sent_col:
    sent_clicked = st.button("分析情感 / Analyse Sentiment", use_container_width=True)

if sent_clicked and st.session_state.sentiment_input.strip():
    sentence = st.session_state.sentiment_input.strip()
    prompt = (
        "What is the sentiment of this financial news sentence? "
        f"Sentence: '{sentence}' "
        "Answer in one word: positive, negative, or neutral."
    )
    with st.spinner("sentiment_sft 推理中… / Analysing…"):
        raw, spd = generate(model_sentiment, tokenizer, prompt)
    pred = raw.strip().lower().split()[0] if raw.strip() else "neutral"
    pred = pred.strip(".,!?")
    if pred not in ("positive", "negative", "neutral"):
        pred = "neutral"
    emoji = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}.get(pred, "⚪")
    st.session_state.sentiment_result = f"{emoji} {pred.upper()}　({spd:.1f} tokens/s)"

if st.session_state.sentiment_result:
    st.markdown(f"""
    <div class="answer-panel">
      <div class="answer-titlebar-finance">🧠 SENTIMENT_SFT · 情感分析结果 / Sentiment Result</div>
      <div class="answer-body" style="font-size:18px;font-weight:bold;">{st.session_state.sentiment_result}</div>
    </div>
    """, unsafe_allow_html=True)

# ── 状态栏 ───────────────────────────────────────────────────
st.markdown("""
<div class="win-statusbar">
  <div class="win-status-panel">
    AutoDL · RTX 4090D · PyTorch 2.1 / CUDA 12.1 &nbsp;|&nbsp;
    Pretrain → General SFT → Finance Fine-tune &nbsp;/&nbsp; 预训练 → 通用SFT → 财务微调
  </div>
</div>
""", unsafe_allow_html=True)
