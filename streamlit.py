# app_streamlit_neural.py
# Streamlit app dengan 3 engine: TF-IDF (cepat), Neural (MiniLM+BiLSTM+Attention), Hybrid (rata2)

import os
import re
import numpy as np
import pandas as pd
import streamlit as st

# === NLP klasik ===
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === Viz ===
import plotly.graph_objects as go

# === Readability (opsional) ===
try:
    from textstat import flesch_reading_ease, flesch_kincaid_grade
except Exception:
    flesch_reading_ease = None
    flesch_kincaid_grade = None

# === Deep learning / HF ===
import tensorflow as tf
import tf_keras as keras
from tf_keras import layers
from transformers import AutoTokenizer, AutoConfig, TFAutoModel

# ---------------------- Page config & CSS ----------------------
st.set_page_config(
    page_title="📝 Advanced Text Similarity Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem; border-radius: 10px; border-left: 5px solid #667eea; margin: 0.5rem 0; }
    .similarity-high { background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); border-left-color: #28a745; }
    .similarity-medium { background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); border-left-color: #ffc107; }
    .similarity-low { background: linear-gradient(135deg, #f8d7da 0%, #f1c0c7 100%); border-left-color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# ---------------------- Util ----------------------
def count_words(text: str) -> int:
    return len(str(text).split())

def estimate_pages(text: str, words_per_page=250) -> float:
    return round(count_words(text) / words_per_page, 1)

def validate_document_length(text: str, max_pages=5):
    pages = estimate_pages(text)
    wc = count_words(text)
    if pages > max_pages:
        return False, f"❌ Dokumen terlalu panjang! ({pages} halaman, {wc} kata). Maks {max_pages} halaman (~{max_pages*250} kata)."
    return True, f"✅ Dokumen valid ({pages} halaman, {wc} kata)"

def get_text_statistics(text: str):
    word_count = count_words(text)
    char_count = len(text)
    sentence_count = len(re.findall(r'[.!?]+', text))
    paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
    if flesch_reading_ease and flesch_kincaid_grade:
        try:
            fr = flesch_reading_ease(text)
            fk = flesch_kincaid_grade(text)
        except Exception:
            fr, fk = 0.0, 0.0
    else:
        fr, fk = 0.0, 0.0
    return {
        "words": word_count,
        "characters": char_count,
        "sentences": sentence_count,
        "paragraphs": paragraph_count,
        "pages": estimate_pages(text),
        "flesch_score": fr,
        "fk_grade": fk
    }

# ---------------------- TF-IDF Engine ----------------------
def calculate_text_similarity_tfidf(text1: str, text2: str):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    mat = vectorizer.fit_transform([text1, text2])
    sim = float(cosine_similarity(mat[0:1], mat[1:2])[0][0])
    feat = vectorizer.get_feature_names_out()

    d1 = mat[0].toarray()[0]
    d2 = mat[1].toarray()[0]
    idx1 = d1.argsort()[-10:][::-1]
    idx2 = d2.argsort()[-10:][::-1]
    doc1_kw = [(feat[i], d1[i]) for i in idx1 if d1[i] > 0]
    doc2_kw = [(feat[i], d2[i]) for i in idx2 if d2[i] > 0]

    return {
        "similarity": sim,
        "doc1_keywords": doc1_kw,
        "doc2_keywords": doc2_kw
    }

# ---------------------- Neural Engine (MiniLM+BiLSTM+Attn) ----------------------
@st.cache_resource(show_spinner=True)
def load_neural_model(
    artifacts_dir="artifacts",
    model_weights="S-001_best_sts.weights.h5",
    model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
    max_len=60,
    bilstm_units=64, attn_units=32,
    d1=64, d2=32, d3=16, dropout=0.3
):
    # Tokenizer: prefer yang disave saat training
    tok_path = os.path.join(artifacts_dir, "tokenizer")
    if os.path.isdir(tok_path):
        tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    config = AutoConfig.from_pretrained(model_name, output_hidden_states=False, output_attentions=False)
    bert = TFAutoModel.from_pretrained(model_name, config=config, from_pt=True)
    bert.trainable = False

    class STSModel(keras.Model):
        def __init__(self, bert_encoder):
            super().__init__()
            self.bert = bert_encoder
            self.ln = layers.LayerNormalization(epsilon=1e-6, name="ln_after_bert")
            self.bilstm = layers.Bidirectional(layers.LSTM(bilstm_units, return_sequences=True), name="bilstm")
            self.attn_tanh = layers.Dense(attn_units, activation="tanh", name="attn_tanh")
            self.attn_score = layers.Dense(1, name="attn_score")
            self.softmax = layers.Softmax(axis=1, name="attn_softmax")
            self.mul = layers.Multiply(name="attn_weighted")
            self.d1 = layers.Dense(d1, activation="relu", name="dense_hidden1")
            self.d2 = layers.Dense(d2, activation="relu", name="dense_hidden2")
            self.d3 = layers.Dense(d3, activation="relu", name="dense_hidden3")
            self.do = layers.Dropout(dropout)
            self.out = layers.Dense(1, activation="sigmoid", name="similarity")

        def call(self, inputs, training=False):
            ids = inputs["input_ids"]; msk = inputs["attention_mask"]; seg = inputs["token_type_ids"]
            x = self.bert(input_ids=ids, attention_mask=msk, token_type_ids=seg, training=False).last_hidden_state
            x = self.ln(x)
            m_bool = tf.cast(msk, tf.bool)
            x = self.bilstm(x, mask=m_bool)
            score = self.attn_score(self.attn_tanh(x))
            mask3 = tf.expand_dims(m_bool, -1)
            neg_inf = tf.fill(tf.shape(score), tf.float32.min)
            score = tf.where(mask3, score, neg_inf)
            alphas = self.softmax(score)
            context = tf.reduce_sum(self.mul([x, alphas]), axis=1)
            h = self.d1(context); h = self.do(h, training=training)
            h = self.d2(h);       h = self.do(h, training=training)
            h = self.d3(h);       h = self.do(h, training=training)
            return self.out(h)

    model = STSModel(bert)

    # build dummy
    dummy = tokenizer("a", "b", padding="max_length", truncation=True, max_length=max_len,
                      return_tensors="tf", return_attention_mask=True, return_token_type_ids=True)
    feats = {
        "input_ids": dummy["input_ids"],
        "attention_mask": dummy["attention_mask"],
        "token_type_ids": dummy.get("token_type_ids", tf.zeros_like(dummy["input_ids"]))
    }
    _ = model(feats, training=False)

    weight_path = os.path.join(artifacts_dir, model_weights)
    if not os.path.isfile(weight_path):
        st.warning(f"⚠️ Weights tidak ditemukan di {weight_path}. Pastikan file ada.")
    else:
        model.load_weights(weight_path)

    return model, tokenizer, max_len

def predict_similarity_neural(model, tokenizer, s1: str, s2: str, max_len=60) -> float:
    enc = tokenizer(
        s1, s2, padding="max_length", truncation=True, max_length=max_len,
        return_tensors="tf", return_attention_mask=True, return_token_type_ids=True
    )
    feats = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "token_type_ids": enc.get("token_type_ids", tf.zeros_like(enc["input_ids"]))
    }
    prob = float(model(feats, training=False).numpy().ravel()[0])  # 0..1
    return prob

# ---------------------- Charts ----------------------
def create_similarity_gauge(similarity: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=similarity*100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Similarity Score (%)"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 60], 'color': "yellow"},
                {'range': [60, 80], 'color': "orange"},
                {'range': [80, 100], 'color': "green"}],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}
        }
    ))
    fig.update_layout(height=300)
    return fig

def create_keywords_comparison_chart(doc1_keywords, doc2_keywords):
    d1 = doc1_keywords[:5]
    d2 = doc2_keywords[:5]
    if not d1 and not d2:
        return None
    fig = go.Figure()
    if d1:
        fig.add_trace(go.Bar(name='Dokumen 1', x=[k for k, _ in d1], y=[v for _, v in d1]))
    if d2:
        fig.add_trace(go.Bar(name='Dokumen 2', x=[k for k, _ in d2], y=[v for _, v in d2]))
    fig.update_layout(title='Top Keywords Comparison (TF-IDF Scores)', barmode='group', height=400)
    return fig

# ---------------------- Sidebar ----------------------
st.sidebar.markdown("## 📋 Menu Navigasi")
analysis_type = st.sidebar.radio("Pilih jenis analisis:", ["📄 Text Similarity", "📁 Document Similarity"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("## ⚙️ Engine")
engine = st.sidebar.selectbox(
    "Metode similarity:",
    ["TF-IDF (cepat)", "Neural (MiniLM+BiLSTM)", "Hybrid (rata-rata)"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Batasan Dokumen:**
- Maks 5 halaman (~1.250 kata)
- Format .txt (untuk upload) atau input manual

**Skor:**
- Neural = probabilitas duplikat/parafrase (0..1)
- TF-IDF = cosine similarity (0..1)
- Hybrid = rata-rata keduanya
""")

# ---------------------- Header ----------------------
st.markdown("""
<div class="main-header">
    <h1>🔍 Advanced Text Similarity Analyzer</h1>
    <p>Analisis kemiripan teks & dokumen (TF-IDF vs Neural MiniLM+BiLSTM+Attention)</p>
</div>
""", unsafe_allow_html=True)

# ---------------------- Main: Text Similarity ----------------------
if analysis_type == "📄 Text Similarity":
    st.markdown("## 📝 Masukkan Teks")
    c1, c2 = st.columns(2)
    with c1:
        t1 = st.text_area("Teks 1", height=200, placeholder="Paste atau ketik teks pertama...")
        if t1:
            ok, msg = validate_document_length(t1)
            st.success(msg) if ok else st.error(msg)
            if not ok: t1 = ""
    with c2:
        t2 = st.text_area("Teks 2", height=200, placeholder="Paste atau ketik teks kedua...")
        if t2:
            ok, msg = validate_document_length(t2)
            st.success(msg) if ok else st.error(msg)
            if not ok: t2 = ""

    if st.button("🚀 Analisis Similarity", type="primary", use_container_width=True):
        if t1 and t2:
            with st.spinner("🔄 Menghitung..."):
                sim = 0.0
                doc1_kw = []; doc2_kw = []

                if engine == "TF-IDF (cepat)":
                    res = calculate_text_similarity_tfidf(t1, t2)
                    sim = res["similarity"]; doc1_kw = res["doc1_keywords"]; doc2_kw = res["doc2_keywords"]

                elif engine == "Neural (MiniLM+BiLSTM)":
                    model, tok, max_len = load_neural_model(
                        artifacts_dir="artifacts",
                        model_weights="S-001_best_sts.weights.h5",
                        model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
                        max_len=60, bilstm_units=64, attn_units=32, d1=64, d2=32, d3=16, dropout=0.3
                    )
                    sim = predict_similarity_neural(model, tok, t1, t2, max_len)

                else:  # Hybrid
                    res = calculate_text_similarity_tfidf(t1, t2)
                    tfidf_s = res["similarity"]
                    doc1_kw = res["doc1_keywords"]; doc2_kw = res["doc2_keywords"]
                    model, tok, max_len = load_neural_model(artifacts_dir="artifacts", model_weights="S-001_best_sts.weights.h5")
                    neural_s = predict_similarity_neural(model, tok, t1, t2, max_len)
                    sim = (tfidf_s + neural_s) / 2.0

            # === Output ===
            st.markdown("## 📊 Hasil")
            cc1, cc2, cc3 = st.columns([1,2,1])
            with cc1:
                klass = "similarity-low"
                label = "🔴 Tidak Mirip"
                if sim >= 0.8: klass, label = "similarity-high", "🟢 Sangat Mirip"
                elif sim >= 0.6: klass, label = "similarity-medium", "🟡 Cukup Mirip"
                st.markdown(f"""
                <div class="metric-card {klass}">
                    <h3>Similarity Score</h3>
                    <h1>{sim:.4f}</h1>
                    <p>({sim*100:.2f}%)</p>
                    <p><strong>{label}</strong></p>
                </div>""", unsafe_allow_html=True)
            with cc2:
                st.plotly_chart(create_similarity_gauge(sim), use_container_width=True)
            with cc3:
                st.metric("Confidence", f"{sim*100:.1f}%")
                st.progress(sim)

            # Keywords hanya relevan untuk TF-IDF / Hybrid
            if (engine != "Neural (MiniLM+BiLSTM)") and (doc1_kw or doc2_kw):
                st.markdown("## 🔤 Keywords (TF-IDF)")
                fig_kw = create_keywords_comparison_chart(doc1_kw, doc2_kw)
                if fig_kw: st.plotly_chart(fig_kw, use_container_width=True)

            # Statistik
            s1 = get_text_statistics(t1); s2 = get_text_statistics(t2)
            colA, colB = st.columns(2)
            with colA:
                st.markdown("### 📈 Statistik Teks 1")
                for k, v in s1.items():
                    if k == "flesch_score": st.metric("Flesch Reading Score", f"{v:.1f}")
                    elif k == "fk_grade":   st.metric("FK Grade Level", f"{v:.1f}")
                    else: st.metric(k.title(), v)
            with colB:
                st.markdown("### 📈 Statistik Teks 2")
                for k, v in s2.items():
                    if k == "flesch_score": st.metric("Flesch Reading Score", f"{v:.1f}")
                    elif k == "fk_grade":   st.metric("FK Grade Level", f"{v:.1f}")
                    else: st.metric(k.title(), v)
        else:
            st.error("❌ Mohon masukkan kedua teks yang valid!")

# ---------------------- Main: Document Similarity ----------------------
else:
    st.markdown("## 📁 Upload 2 Dokumen (.txt)")
    c1, c2 = st.columns(2)
    with c1:
        f1 = st.file_uploader("Dokumen 1", type="txt", key="file1")
        doc1 = f1.read().decode("utf-8") if f1 else ""
        if doc1:
            ok, msg = validate_document_length(doc1)
            st.success(msg) if ok else st.error(msg)
            if ok:
                with st.expander("👁️ Preview 1"):
                    st.text_area("Content:", doc1[:500] + ("..." if len(doc1) > 500 else ""), height=150, disabled=True)
            else:
                doc1 = ""
    with c2:
        f2 = st.file_uploader("Dokumen 2", type="txt", key="file2")
        doc2 = f2.read().decode("utf-8") if f2 else ""
        if doc2:
            ok, msg = validate_document_length(doc2)
            st.success(msg) if ok else st.error(msg)
            if ok:
                with st.expander("👁️ Preview 2"):
                    st.text_area("Content:", doc2[:500] + ("..." if len(doc2) > 500 else ""), height=150, disabled=True)
            else:
                doc2 = ""

    if st.button("🔍 Bandingkan Dokumen", type="primary", use_container_width=True):
        if doc1 and doc2:
            with st.spinner("📊 Menganalisis..."):
                sim = 0.0
                doc1_kw = []; doc2_kw = []

                if engine == "TF-IDF (cepat)":
                    res = calculate_text_similarity_tfidf(doc1, doc2)
                    sim = res["similarity"]; doc1_kw = res["doc1_keywords"]; doc2_kw = res["doc2_keywords"]

                elif engine == "Neural (MiniLM+BiLSTM)":
                    model, tok, max_len = load_neural_model(
                        artifacts_dir="artifacts",
                        model_weights="S-001_best_sts.weights.h5",
                        model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
                        max_len=60, bilstm_units=64, attn_units=32, d1=64, d2=32, d3=16, dropout=0.3
                    )
                    sim = predict_similarity_neural(model, tok, doc1, doc2, max_len)

                else:
                    res = calculate_text_similarity_tfidf(doc1, doc2)
                    tfidf_s = res["similarity"]
                    doc1_kw = res["doc1_keywords"]; doc2_kw = res["doc2_keywords"]
                    model, tok, max_len = load_neural_model(artifacts_dir="artifacts", model_weights="S-001_best_sts.weights.h5")
                    neural_s = predict_similarity_neural(model, tok, doc1, doc2, max_len)
                    sim = (tfidf_s + neural_s)/2.0

            st.markdown("## 🎯 Hasil")
            cA, cB, cC, cD = st.columns(4)
            with cA: st.metric("Similarity", f"{sim:.4f}")
            with cB: st.metric("Percentage", f"{sim*100:.2f}%")
            s1 = get_text_statistics(doc1); s2 = get_text_statistics(doc2)
            with cC: st.metric("Doc 1 Pages", s1["pages"])
            with cD: st.metric("Doc 2 Pages", s2["pages"])
            st.plotly_chart(create_similarity_gauge(sim), use_container_width=True)

            if (engine != "Neural (MiniLM+BiLSTM)") and (doc1_kw or doc2_kw):
                st.plotly_chart(create_keywords_comparison_chart(doc1_kw, doc2_kw), use_container_width=True)

            st.markdown("## 📋 Perbandingan Detail")
            comp = pd.DataFrame({
                "Metric": ["Words","Characters","Sentences","Paragraphs","Pages","Flesch Score","FK Grade"],
                "Dokumen 1": [s1["words"], s1["characters"], s1["sentences"], s1["paragraphs"],
                              s1["pages"], f"{s1['flesch_score']:.1f}", f"{s1['fk_grade']:.1f}"],
                "Dokumen 2": [s2["words"], s2["characters"], s2["sentences"], s2["paragraphs"],
                              s2["pages"], f"{s2['flesch_score']:.1f}", f"{s2['fk_grade']:.1f}"]
            })
            st.dataframe(comp, use_container_width=True)
        else:
            st.error("❌ Mohon upload dua dokumen yang valid.")

# ---------------------- Footer ----------------------
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#666; padding:1rem;'>
  <p>🎓 <strong>Advanced Text Similarity Analyzer</strong> | Versi TF-IDF & Neural</p>
  <p>⚡ Streamlit • Scikit-learn • Transformers • TensorFlow</p>
</div>
""", unsafe_allow_html=True)
