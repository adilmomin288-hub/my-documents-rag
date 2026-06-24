import os
import pickle
import pandas as pd
import numpy as np
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from sklearn.feature_extraction.text import TfidfVectorizer

class LocalEmbeddings:
    def __init__(self):
        self.vectorizer = None

    def embed_documents(self, texts):
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(stop_words="english")
            vectors = self.vectorizer.fit_transform(texts)
        else:
            vectors = self.vectorizer.transform(texts)
        return vectors.astype(np.float32).toarray().tolist()

    def embed_query(self, text):
        if self.vectorizer is None:
            raise ValueError("Vectorizer is not fitted. Run embed_documents first.")
        vector = self.vectorizer.transform([text])
        return vector.astype(np.float32).toarray()[0].tolist()

class SimpleTextSplitter:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text):
        text = text.strip()
        if not text:
            return []
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end].strip())
            if end == len(text):
                break
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def split_documents(self, documents):
        new_docs = []
        for document in documents:
            for chunk in self.split_text(document.page_content):
                new_docs.append(Document(page_content=chunk))
        return new_docs

csv_path = os.path.join("data", "IMDB-Movie-Data.csv")
df = pd.read_csv(csv_path)

documents = []

for _, row in df.iterrows():

    content = f"""
Title: {row['Title']}
Genre: {row['Genre']}
Description: {row['Description']}
Director: {row['Director']}
Actors: {row['Actors']}
Year: {row['Year']}
Rating: {row['Rating']}
"""

    documents.append(
        Document(page_content=content)
    )

splitter = SimpleTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)

embeddings = LocalEmbeddings()

db = FAISS.from_documents(
    docs,
    embeddings
)

os.makedirs("vector_store", exist_ok=True)
vectorizer_path = os.path.join("vector_store", "tfidf_vectorizer.pkl")
with open(vectorizer_path, "wb") as f:
    pickle.dump(embeddings.vectorizer, f)

db.save_local(
    "vector_store/faiss_index"
)

print("FAISS Index Created Successfully")
print(f"Vectorizer saved to {vectorizer_path}")