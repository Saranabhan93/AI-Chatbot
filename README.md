# AI Assistant with RAG (PDF Q&A) 🚀

A Retrieval-Augmented Generation (RAG) pipeline built with Python, Streamlit, LangChain, and FAISS, capable of answering questions accurately using external PDF documents.

## 📌 Project Overview
This application allows users to upload a PDF document, extract and chunk its text, generate embeddings using HuggingFace models, store them in a FAISS vector database, and perform context-aware question answering powered by LLM APIs (Gemini).

## 🏗️ Architecture Diagram
The pipeline follows a structured RAG workflow:
1. **PDF Upload**: User selects and uploads a PDF file via the Streamlit UI.
2. **Text Extraction**: Extracts text page-by-page using `pdfplumber`.
3. **Text Splitting**: Breaks down the text into manageable chunks.
4. **Embeddings**: Generates vector embeddings using `HuggingFace` (`all-MiniLM-L6-v2`).
5. **Vector Store**: Indexes and stores embeddings locally using `FAISS`.
6. **Retrieval**: Searches for the top matching chunks based on user queries.
7. **Generation**: Sends the retrieved context and user prompt to Gemini to generate an accurate, source-grounded response.

## 🛠️ Technologies & Frameworks Used
* **Python 3.10+**
* **Streamlit** (Web UI)
* **LangChain** (RAG Orchestration)
* **FAISS** (Vector Database / Indexing)
* **HuggingFace Embeddings** (`all-MiniLM-L6-v2`)
* **Google Gemini API** (LLM for Answer Generation)
* **pdfplumber** (PDF text extraction)

## ⚙️ Installation & Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Saranabhan93/RAG-Assistant.git](https://github.com/Saranabhan93/RAG-Assistant.git)
   cd RAG-Assistant