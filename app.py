import os
import pickle
import streamlit as st
from langchain_community.vectorstores import FAISS
from typing import List

VECTOR_PATH = os.path.join("vector_store", "tfidf_vectorizer.pkl")

class LocalEmbeddings:
    def __init__(self):
        if not os.path.exists(VECTOR_PATH):
            raise FileNotFoundError(
                "TF-IDF vectorizer not found. Run ingest.py first to create vector_store."
            )
        with open(VECTOR_PATH, "rb") as f:
            self.vectorizer = pickle.load(f)

    def embed_documents(self, texts: List[str]):
        if self.vectorizer is None:
            raise ValueError("Vectorizer is not loaded. Run ingest.py first.")
        vectors = self.vectorizer.transform(texts)
        return vectors.astype("float32").toarray().tolist()

    def embed_query(self, text: str):
        if self.vectorizer is None:
            raise ValueError("Vectorizer is not loaded. Run ingest.py first.")
        vector = self.vectorizer.transform([text])
        return vector.astype("float32").toarray()[0].tolist()

    def __call__(self, text):
        if isinstance(text, list):
            return self.embed_documents(text)
        return self.embed_query(text)


def build_answer(question: str, docs):
    if not docs:
        return "I couldn't find matching movie information. Try another question."

    question_lower = question.lower()
    if "director" in question_lower or "directed" in question_lower:
        return "\n\n".join(
            [doc.page_content for doc in docs if "Director:" in doc.page_content][:3]
        ) or "I found relevant movie entries, but no director line was available."

    if "genre" in question_lower:
        return "\n\n".join(
            [doc.page_content for doc in docs if "Genre:" in doc.page_content][:3]
        ) or "I found relevant movie entries, but no genre line was available."

    if "rating" in question_lower or "rate" in question_lower:
        return "\n\n".join(
            [doc.page_content for doc in docs if "Rating:" in doc.page_content][:3]
        ) or "I found relevant movie entries, but no rating line was available."

    return (
        "I found these relevant movie entries based on your question:\n\n"
        + "\n\n".join([doc.page_content for doc in docs[:3]])
    )

st.set_page_config(
    page_title="Movie Chatbot",
    page_icon="🎬"
)

st.title("🎬 Movie Chatbot")

embeddings = LocalEmbeddings()

db = FAISS.load_local(
    "vector_store/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

question = st.text_input("Ask About Movies")

if question:

    docs = db.similarity_search(question, k=3)
    answer = build_answer(question, docs)
    st.write(answer)