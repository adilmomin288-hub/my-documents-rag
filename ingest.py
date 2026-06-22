import pandas as pd

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from sentence_transformers import SentenceTransformer


class LocalEmbeddings:

    def __init__(self):

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def embed_documents(self, texts):

        return self.model.encode(texts).tolist()

    def embed_query(self, text):

        return self.model.encode(text).tolist()


csv_file = "data/IMDB-Movie-Data.csv"

df = pd.read_csv(csv_file)

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

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)

embeddings = LocalEmbeddings()

db = FAISS.from_documents(
    docs,
    embeddings
)

db.save_local(
    "vector_store/faiss_index"
)

print("FAISS Index Created Successfully")