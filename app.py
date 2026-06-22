import os
import streamlit as st
from dotenv import load_dotenv

from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HF_TOKEN:
    st.error("HUGGINGFACEHUB_API_TOKEN not found in .env file")
    st.stop()

# -------------------------------
# Hugging Face Client
# -------------------------------
client = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta", token=HF_TOKEN)

# -------------------------------
# Embeddings model
# -------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------
# Load FAISS Database
# -------------------------------
try:
    db = FAISS.load_local(
        "vector_store/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
except Exception as e:
    st.error(f"Error loading FAISS index: {e}")
    st.stop()

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(
    page_title="Movie Chatbot",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Question Answer Chatbot")
st.write("Ask anything about movies.")

question = st.text_input("Enter your question:")

if st.button("Get Answer"):

    if question.strip() == "":
        st.warning("Please enter a question.")

    else:
        docs = db.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are an expert Movie Assistant.

Use only the context below.

Context:
{context}

Question:
{question}

Give a detailed and accurate answer.
"""

        try:
            # ✅ Use text_generation instead of conversational
            response = client.text_generation(
                prompt,
                max_new_tokens=300,
                temperature=0.3,
            )

            # ✅ Response is already plain text
            st.success("Answer")
            st.info(response)

        except Exception as e:
            st.error("⚠️ Something went wrong while generating the answer.")
            st.text(str(e))
