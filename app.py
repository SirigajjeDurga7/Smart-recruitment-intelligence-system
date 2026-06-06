import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
import matplotlib.pyplot as plt

st.set_page_config(page_title="Recruitment Dashboard", layout="wide")

# -----------------------------
# PDF TEXT EXTRACTION
# -----------------------------
def extract_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.lower()

# -----------------------------
# POSITIONAL ENCODING
# -----------------------------
def positional_encoding(max_len, d_model=16):
    PE = np.zeros((max_len, d_model))
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            angle = pos / np.power(10000, (2 * (i//2)) / d_model)
            PE[pos, i] = np.sin(angle)
            if i+1 < d_model:
                PE[pos, i+1] = np.cos(angle)
    return PE

# -----------------------------
# SIDEBAR INPUT
# -----------------------------
st.title("📄 AI Recruitment Dashboard")

jd = st.text_area("📌 Enter Job Description")

uploaded_files = st.file_uploader(
    "Upload Resumes (PDF)", 
    type=["pdf"], 
    accept_multiple_files=True
)

# -----------------------------
# PROCESS BUTTON
# -----------------------------
if st.button("🚀 Rank Candidates"):

    if not jd or not uploaded_files:
        st.warning("Please upload JD and resumes")
        st.stop()

    # Extract resume text
    resumes = []
    names = []

    for file in uploaded_files:
        text = extract_text(file)
        resumes.append(text)
        names.append(file.name)

    # TF-IDF Ranking
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(resumes + [jd])

    resume_vec = vectors[:-1]
    jd_vec = vectors[-1]

    scores = cosine_similarity(jd_vec, resume_vec).flatten()

    df = pd.DataFrame({
        "Candidate": names,
        "Score": scores
    }).sort_values(by="Score", ascending=False)

    st.subheader("🏆 Ranked Candidates")
    st.dataframe(df)

    # -----------------------------
    # DOWNLOAD RESULTS
    # -----------------------------
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇ Download Results CSV", csv, "ranking.csv", "text/csv")

    # -----------------------------
    # TOP CANDIDATE
    # -----------------------------
    top_resume = resumes[df.index[0]]

    st.subheader("📌 Top Candidate Insights")

    st.write("Top Skills (TF-IDF Proxy):")
    feature_names = vectorizer.get_feature_names_out()
    top_vec = resume_vec[df.index[0]].toarray().flatten()

    top_words = [feature_names[i] for i in top_vec.argsort()[-10:]]
    st.write(top_words)

    # -----------------------------
    # POSITIONAL ENCODING HEATMAP
    # -----------------------------
    st.subheader("📊 Positional Encoding Heatmap")

    tokens = top_resume.split()[:40]
    pe = positional_encoding(len(tokens), 16)

    fig, ax = plt.subplots()
    ax.imshow(pe, aspect='auto')
    ax.set_title("Positional Encoding Matrix")
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Token Position")
    st.pyplot(fig)

    # -----------------------------
    # SIMPLE ATTENTION HEATMAP (SIMULATED)
    # -----------------------------
    st.subheader("🔥 Attention Heatmap (Simulated)")

    attn = np.random.rand(len(tokens), len(tokens))

    fig2, ax2 = plt.subplots()
    ax2.imshow(attn)
    ax2.set_title("Self-Attention Heatmap")
    st.pyplot(fig2)

    st.success("Analysis Complete 🚀")