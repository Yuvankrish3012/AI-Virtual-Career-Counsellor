import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="🎯 AI Career Counsellor",
    layout="centered",
    page_icon="🎓"
)

# --- TITLE & INFO ---
st.title("🎯 AI Virtual Career Counsellor")
st.markdown("Ask your career-related questions and get instant suggestions using NLP-powered AI 💬")
st.divider()

# --- LOAD DATA & MODEL ---
@st.cache_resource
def load_model_and_data():
    model = SentenceTransformer(r"D:\ML PROJECTS\AI Virtual Career Counsellor\career_embedding_model")
    df = pickle.load(open(r"D:\ML PROJECTS\AI Virtual Career Counsellor\career_df.pkl", "rb"))
    embeddings = np.load(r"D:\ML PROJECTS\AI Virtual Career Counsellor\career_embeddings.npy")
    return model, df, embeddings

model, df, corpus_embeddings = load_model_and_data()
corpus_questions = df['question'].tolist()
corpus_answers = df['answer'].tolist()

# --- INPUT ---
user_question = st.text_input("📌 Enter your career question below:", placeholder="E.g., What should I do after B.Sc Computer Science?")

# --- SEARCH FUNCTION ---
def search_answer(query, top_k=3):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    return [(corpus_questions[i], corpus_answers[i], similarities[i]) for i in top_k_indices]

# --- RESPONSE ---
if user_question:
    st.markdown("### 🧠 Best Career Suggestions:")
    with st.spinner("Analyzing your query..."):
        results = search_answer(user_question, top_k=3)

    for i, (q, a, score) in enumerate(results, 1):
        st.markdown(f"**{i}. {a}**")
        with st.expander("See related query the AI matched with"):
            st.markdown(f"🗨️ **Similar Question:** _{q}_ \n\n🔎 **Similarity Score:** `{score:.4f}`")

    st.success("✅ Suggestions generated!")
