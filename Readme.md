# 🎬 Movie Question Answer Chatbot

A Movie Question Answer Chatbot built using **LangChain, FAISS, Hugging Face, Streamlit, and Sentence Transformers**. This project uses Retrieval-Augmented Generation (RAG) to answer movie-related questions from the IMDB Movie Dataset.

## 🚀 Features

* Movie Question Answering
* Retrieval-Augmented Generation (RAG)
* Semantic Search using FAISS
* Hugging Face LLM Integration
* Interactive Streamlit UI
* IMDB Movie Dataset Support
* Fast and Accurate Movie Recommendations
* Natural Language Responses
* Free and Open Source

---

## 🛠️ Technologies Used

* Python
* LangChain
* FAISS Vector Database
* Hugging Face Inference API
* Sentence Transformers
* Streamlit
* Pandas
* IMDB Movie Dataset

---

## 📂 Project Structure

Movie-Chatbot/

├── data/

│ └── IMDB-Movie-Data.csv

├── vector_store/

│ └── faiss_index

├── app.py

├── ingest.py

├── requirements.txt

├── .env

└── README.md

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/movie-chatbot.git

cd movie-chatbot
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

Windows

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file and add:

```env
HUGGINGFACEHUB_API_TOKEN=YOUR_HUGGINGFACE_TOKEN
```

---

## 📊 Build Vector Database

Run:

```bash
python ingest.py
```

Output:

```text
FAISS Index Created Successfully
```

---

## ▶️ Run Application

```bash
streamlit run app.py
```

Application will start at:

```text
http://localhost:8501
```

---

## 💬 Example Questions

* Tell me about Interstellar
* Who directed Inception?
* Recommend thriller movies
* Best action movies
* Movies starring Leonardo DiCaprio
* Top rated sci-fi movies
* Best drama movies
* Movies released in 2016

---

## 🧠 How It Works

1. IMDB dataset is loaded.
2. Text data is converted into embeddings.
3. Embeddings are stored in FAISS.
4. User asks a question.
5. Relevant movie information is retrieved.
6. Hugging Face LLM generates a natural language response.
7. Streamlit displays the answer.

---

## 📈 Future Enhancements

* Movie Posters Integration
* User Authentication
* Chat History
* Genre-Based Recommendations
* Voice Assistant Support
* Multi-Language Support
* Advanced LLM Models

---

## 🎯 Learning Outcomes

* Retrieval-Augmented Generation (RAG)
* Vector Databases
* LangChain Framework
* Semantic Search
* LLM Integration
* Streamlit Deployment
* End-to-End AI Application Development

---

## 👨‍💻 Author

Adil Momin

GitHub: https://github.com/adilmomin288-hub

---

## ⭐ Support

If you found this projectdel venv\.gitignore useful, please give it a star on GitHub and share it with others.

Thank You!