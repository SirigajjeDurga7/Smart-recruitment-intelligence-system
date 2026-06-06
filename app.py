import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
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
            angle = pos / np.power(10000, (2 * (i // 2)) / d_model)
            PE[pos, i] = np.sin(angle)
            if i + 1 < d_model:
                PE[pos, i + 1] = np.cos(angle)
    return PE

# -----------------------------
# UI
# -----------------------------
st.title("📄 AI Recruitment Dashboard")

jd = st.text_area("📌 Enter Job Description")

uploaded_files = st.file_uploader(
    "Upload Resumes (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

# -----------------------------
# PROCESS
# -----------------------------
if st.button("🚀 Rank Candidates"):

    if not jd.strip() or not uploaded_files:
        st.warning("⚠️ Please upload JD and resumes")
        st.stop()

    resumes = []
    names = []

    # Extract resumes
    for file in uploaded_files:
        text = extract_text(file)
        resumes.append(text)
        names.append(file.name)

    if len(resumes) == 0:
        st.error("No valid resumes found")
        st.stop()

    # -----------------------------
    # TF-IDF RANKING
    # -----------------------------
    vectorizer = TfidfVectorizer(stop_words="english")

    docs = resumes + [jd]
    vectors = vectorizer.fit_transform(docs)

    resume_vec = vectors[:-1]
    jd_vec = vectors[-1]

    scores = cosine_similarity(jd_vec, resume_vec).flatten()

    df = pd.DataFrame({
        "Candidate": names,
        "Score": scores
    }).sort_values(by="Score", ascending=False).reset_index(drop=True)

    # -----------------------------
    # OUTPUT TABLE
    # -----------------------------
    st.subheader("🏆 Ranked Candidates")
    st.dataframe(df, use_container_width=True)

    # -----------------------------
    # DOWNLOAD
    # -----------------------------
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download Results CSV", csv, "ranking.csv", "text/csv")

    # -----------------------------
    # TOP CANDIDATE
    # -----------------------------
    st.subheader("📌 Top Candidate Insights")

    top_index = 0
    top_resume = resumes[top_index]

    feature_names = vectorizer.get_feature_names_out()

    top_vector = resume_vec[top_index].toarray().flatten()

    top_indices = top_vector.argsort()[-10:][::-1]
    top_words = [feature_names[i] for i in top_indices if top_vector[i] > 0]

    st.write("🔥 Top Skills (TF-IDF Extracted):")
    st.write(top_words)

    # -----------------------------
    # POSITIONAL ENCODING
    # -----------------------------
    st.subheader("📊 Positional Encoding Heatmap")

    tokens = top_resume.split()[:40]
    pe = positional_encoding(len(tokens), 16)

    fig, ax = plt.subplots()
    ax.imshow(pe, aspect="auto")
    ax.set_title("Positional Encoding Matrix")
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Token Position")
    st.pyplot(fig)

    # -----------------------------
    # ATTENTION HEATMAP (SIMULATED)
    # -----------------------------
    st.subheader("🔥 Attention Heatmap (Simulated)")

    attn = np.random.rand(len(tokens), len(tokens))

    fig2, ax2 = plt.subplots()
    ax2.imshow(attn)
    ax2.set_title("Self-Attention Heatmap")
    st.pyplot(fig2)

    st.success("✅ Analysis Complete")
