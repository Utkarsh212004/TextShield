import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

st.set_page_config(
    page_title="TextShield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

/* Hide streamlit default elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #1a1a2e 100%);
    border-right: 1px solid rgba(99, 102, 241, 0.2);
}

/* Cards */
.shield-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 16px;
    padding: 24px;
    margin: 12px 0;
    backdrop-filter: blur(10px);
}

/* Hero section */
.hero-section {
    text-align: center;
    padding: 2rem 0 1rem 0;
}
.hero-title {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #c084fc, #fb7185);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}
.hero-subtitle {
    color: rgba(255,255,255,0.5);
    font-size: 1.1rem;
    font-weight: 300;
}

/* Result cards */
.result-fake {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(220,38,38,0.05));
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    margin: 16px 0;
}
.result-real {
    background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(22,163,74,0.05));
    border: 1px solid rgba(34,197,94,0.4);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    margin: 16px 0;
}
.result-spam {
    background: linear-gradient(135deg, rgba(251,146,60,0.15), rgba(234,88,12,0.05));
    border: 1px solid rgba(251,146,60,0.4);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    margin: 16px 0;
}
.result-ham {
    background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(22,163,74,0.05));
    border: 1px solid rgba(34,197,94,0.4);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    margin: 16px 0;
}

.result-icon { font-size: 3rem; margin-bottom: 8px; }
.result-label {
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 4px;
}
.result-fake .result-label { color: #f87171; }
.result-real .result-label { color: #4ade80; }
.result-spam .result-label { color: #fb923c; }
.result-ham .result-label  { color: #4ade80; }
.result-conf {
    font-size: 0.95rem;
    color: rgba(255,255,255,0.5);
}

/* Stat boxes */
.stat-row {
    display: flex;
    gap: 12px;
    margin: 16px 0;
}
.stat-box {
    flex: 1;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.stat-val {
    font-size: 1.5rem;
    font-weight: 600;
    color: #818cf8;
}
.stat-label {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.4);
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Confidence bar */
.conf-bar-bg {
    background: rgba(255,255,255,0.08);
    border-radius: 999px;
    height: 8px;
    margin: 12px 0;
    overflow: hidden;
}
.conf-bar-fill-fake { background: linear-gradient(90deg,#f87171,#ef4444); border-radius:999px; height:8px; }
.conf-bar-fill-real { background: linear-gradient(90deg,#4ade80,#22c55e); border-radius:999px; height:8px; }
.conf-bar-fill-spam { background: linear-gradient(90deg,#fb923c,#f97316); border-radius:999px; height:8px; }
.conf-bar-fill-ham  { background: linear-gradient(90deg,#4ade80,#22c55e); border-radius:999px; height:8px; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid rgba(255,255,255,0.08);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: rgba(255,255,255,0.5);
    font-weight: 500;
    padding: 8px 24px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
}

/* Text area */
.stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 12px !important;
    color: white !important;
    font-size: 0.95rem !important;
    line-height: 1.7 !important;
}
.stTextArea textarea:focus {
    border-color: rgba(99,102,241,0.8) !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.2) !important;
}

/* Button */
.stButton button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 32px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.4) !important;
}

/* Sidebar logo area */
.sidebar-logo {
    text-align: center;
    padding: 1.5rem 0 1rem;
}
.sidebar-logo-icon {
    font-size: 3rem;
    display: block;
    margin-bottom: 8px;
}
.sidebar-logo-title {
    font-size: 1.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.sidebar-logo-sub {
    font-size: 0.8rem;
    color: rgba(255,255,255,0.35);
    margin-top: 2px;
}

/* Sidebar stat */
.sb-stat {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    font-size: 0.82rem;
    color: rgba(255,255,255,0.45);
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.sb-stat span:last-child {
    color: rgba(255,255,255,0.75);
    font-weight: 500;
}

/* Section label */
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(99,102,241,0.8);
    margin: 1.2rem 0 0.5rem;
}

/* Divider */
.custom-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 1rem 0;
}

/* Badge */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.badge-purple {
    background: rgba(99,102,241,0.2);
    color: #a5b4fc;
    border: 1px solid rgba(99,102,241,0.3);
}
.badge-green {
    background: rgba(34,197,94,0.15);
    color: #86efac;
    border: 1px solid rgba(34,197,94,0.3);
}
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────
@st.cache_resource
def load_models():
    fake_vectorizer = joblib.load('models/tfidf_fake_news.pkl')
    fake_model      = joblib.load('models/fake_news_model.pkl')
    spam_vectorizer = joblib.load('models/tfidf_spam.pkl')
    spam_model      = joblib.load('models/spam_model.pkl')
    return fake_vectorizer, fake_model, spam_vectorizer, spam_model

fake_vectorizer, fake_model, spam_vectorizer, spam_model = load_models()

# ── Text cleaning ─────────────────────────────────────────
stop_words = set(stopwords.words('english'))
stemmer    = PorterStemmer()

def clean_text(text):
    text  = text.lower()
    text  = re.sub(r'http\S+|www\S+', '', text)
    text  = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [stemmer.stem(w) for w in words
             if w not in stop_words and len(w) > 2]
    return ' '.join(words)

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span class="sidebar-logo-icon">🛡️</span>
        <div class="sidebar-logo-title">TextShield</div>
        <div class="sidebar-logo-sub">NLP Detection Engine</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Models</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-stat"><span>🗞️ Fake News</span><span>Logistic Regression</span></div>
    <div class="sb-stat"><span>📩 Spam</span><span>Logistic Regression</span></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Performance</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-stat"><span>Fake News Accuracy</span><span>~99%</span></div>
    <div class="sb-stat"><span>Spam Accuracy</span><span>~98%</span></div>
    <div class="sb-stat"><span>Vectorizer</span><span>TF-IDF</span></div>
    <div class="sb-stat"><span>Features</span><span>5,000</span></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Datasets</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-stat"><span>ISOT Articles</span><span>44,898</span></div>
    <div class="sb-stat"><span>SMS Messages</span><span>5,574</span></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Built by</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-stat"><span>Developer</span><span>Utkarsh Jain</span></div>
    
    """, unsafe_allow_html=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;padding:.5rem 0">
        <span class="badge badge-purple">Python</span>&nbsp;
        <span class="badge badge-purple">Scikit-learn</span>&nbsp;
        <span class="badge badge-green">Streamlit</span>
    </div>
    """, unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────
st.markdown("""
<div class="hero-section">
    <div class="hero-title">🛡️ TextShield</div>
    <div class="hero-subtitle">
        AI-powered fake news and spam detection using NLP
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🗞️  Fake News Detector", "📩  Spam Detector"])

# ════════════════════════════════════════════════
# TAB 1 — FAKE NEWS
# ════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="shield-card">', unsafe_allow_html=True)
    st.markdown("#### Paste a news article or headline")
    st.caption("The model will analyze the writing style and vocabulary to determine authenticity.")

    fn_text = st.text_area(
        "News text",
        height=160,
        placeholder="Paste any news article or headline here...",
        key="fn_input",
        label_visibility="collapsed"
    )
    fn_btn = st.button("🔍  Analyze News", key="fn_btn")
    st.markdown('</div>', unsafe_allow_html=True)

    if fn_btn:
        if not fn_text.strip():
            st.warning("Please paste some text first.")
        else:
            with st.spinner("Analyzing..."):
                cleaned    = clean_text(fn_text)
                vector     = fake_vectorizer.transform([cleaned])
                pred       = fake_model.predict(vector)[0]
                proba      = fake_model.predict_proba(vector)[0]
                confidence = round(max(proba) * 100, 1)
                word_count = len(fn_text.split())
                token_count= len(cleaned.split())

            if pred == 0:
                st.markdown(f"""
                <div class="result-fake">
                    <div class="result-icon">🚨</div>
                    <div class="result-label">FAKE NEWS</div>
                    <div class="result-conf">Model is {confidence}% confident this is fake</div>
                    <div class="conf-bar-bg">
                        <div class="conf-bar-fill-fake" style="width:{confidence}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-real">
                    <div class="result-icon">✅</div>
                    <div class="result-label">REAL NEWS</div>
                    <div class="result-conf">Model is {confidence}% confident this is real</div>
                    <div class="conf-bar-bg">
                        <div class="conf-bar-fill-real" style="width:{confidence}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="stat-row">
                <div class="stat-box">
                    <div class="stat-val">{confidence}%</div>
                    <div class="stat-label">Confidence</div>
                </div>
                <div class="stat-box">
                    <div class="stat-val">{word_count}</div>
                    <div class="stat-label">Words input</div>
                </div>
                <div class="stat-box">
                    <div class="stat-val">{token_count}</div>
                    <div class="stat-label">Clean tokens</div>
                </div>
                <div class="stat-box">
                    <div class="stat-val">LR</div>
                    <div class="stat-label">Model used</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("🔬 See cleaned tokens fed to model"):
                st.code(cleaned, language=None)

# ════════════════════════════════════════════════
# TAB 2 — SPAM
# ════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="shield-card">', unsafe_allow_html=True)
    st.markdown("#### Paste an SMS or email message")
    st.caption("The model will classify whether the message is spam or legitimate.")

    sp_text = st.text_area(
        "Message text",
        height=160,
        placeholder="Paste any SMS or email message here...",
        key="sp_input",
        label_visibility="collapsed"
    )
    sp_btn = st.button("🔍  Analyze Message", key="sp_btn")
    st.markdown('</div>', unsafe_allow_html=True)

    if sp_btn:
        if not sp_text.strip():
            st.warning("Please paste some text first.")
        else:
            with st.spinner("Analyzing..."):
                cleaned    = clean_text(sp_text)
                vector     = spam_vectorizer.transform([cleaned])
                pred       = spam_model.predict(vector)[0]
                proba      = spam_model.predict_proba(vector)[0]
                confidence = round(max(proba) * 100, 1)
                word_count = len(sp_text.split())
                token_count= len(cleaned.split())

            if pred == 1:
                st.markdown(f"""
                <div class="result-spam">
                    <div class="result-icon">🚫</div>
                    <div class="result-label">SPAM</div>
                    <div class="result-conf">Model is {confidence}% confident this is spam</div>
                    <div class="conf-bar-bg">
                        <div class="conf-bar-fill-spam" style="width:{confidence}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-ham">
                    <div class="result-icon">✅</div>
                    <div class="result-label">NOT SPAM</div>
                    <div class="result-conf">Model is {confidence}% confident this is legitimate</div>
                    <div class="conf-bar-bg">
                        <div class="conf-bar-fill-ham" style="width:{confidence}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="stat-row">
                <div class="stat-box">
                    <div class="stat-val">{confidence}%</div>
                    <div class="stat-label">Confidence</div>
                </div>
                <div class="stat-box">
                    <div class="stat-val">{word_count}</div>
                    <div class="stat-label">Words input</div>
                </div>
                <div class="stat-box">
                    <div class="stat-val">{token_count}</div>
                    <div class="stat-label">Clean tokens</div>
                </div>
                <div class="stat-box">
                    <div class="stat-val">LR</div>
                    <div class="stat-label">Model used</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("🔬 See cleaned tokens fed to model"):
                st.code(cleaned, language=None)

# ── Footer ────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem;color:rgba(255,255,255,0.2);font-size:0.8rem">
    TextShield · Built with Python, Scikit-learn, NLTK & Streamlit
     Developed and Designed by Utkarsh Jain
</div>
""", unsafe_allow_html=True)