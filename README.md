# Intelligent Document Question-Answering System (RAG)

> A Retrieval-Augmented Generation (RAG) system to answer questions from documents (PDFs / text) using embeddings + LLMs.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Repository Structure](#repository-structure)  
4. [Prerequisites & Setup](#prerequisites--setup)  
5. [Running the System](#running-the-system)  
   - [Indexing Documents](#indexing-documents)  
   - [Querying / Chat](#querying--chat)  
   - [Web / App Interface](#web--app-interface)  
6. [Configuration & Environment Variables](#configuration--environment-variables)  
7. [Database / Persistence](#database--persistence)  
8. [Extending / Customizing](#extending--customizing)  
9. [Troubleshooting & Tips](#troubleshooting--tips)  
10. [Credits & License](#credits--license)  

---

## Project Overview

This repository implements a **Document Question-Answering System** using a *Retrieval-Augmented Generation* (RAG) approach. The system:

- Ingests documents (PDFs / text files),
- Splits them into chunks,
- Builds a vector store / embedding index,
- Retrieves relevant chunks for a user’s query,
- Invokes an LLM to generate an answer using retrieved context + prompt templates.

It can be used as a standalone script, via a notebook demo, or as a small web/chat interface.

---

## Features

- Multi-PDF ingestion & indexing  
- Vector store (persistent) support  
- Query / chat interface  
- (Optional) Web or UI frontend  
- SQL / database support for logging or metadata  
- Notebook demo for stepping through pipeline  

---

## Repository Structure
```
├── Multi_PDF_RAG.py # Core ingestion + query pipeline
├── main.py # Example or wrapper entrypoint
├── final_app.py # UI / app frontend
├── app.py # Alternate frontend / interface
├── chatbot.ipynb # Notebook demo of full pipeline
├── requirements.txt # Python dependencies
├── mysql_db.sql # SQL schema & sample data for MySQL
├── sql.py # DB utilities (SQLite / MySQL)
├── student.db # SQLite DB included (for demo / logging)
├── SGP Presentation.pptx # Project presentation slides
├── attention.pdf # Supplementary reading / resource
└── README.md # This file
```

- **Multi_PDF_RAG.py** — Core logic: load documents, split, embed, index, query  
- **main.py** — Example script / wrapper to try the pipeline  
- **final_app.py** / **app.py** — Frontend interface (Flask, Streamlit, or similar)  
- **chatbot.ipynb** — Interactive demo for experimentation  
- **sql.py** / **mysql_db.sql** / **student.db** — Database setup & utilities  
- **requirements.txt** — All required Python packages  
- **SGP Presentation.pptx**, **attention.pdf** — Supporting docs

---

## Prerequisites & Setup

1. **Clone repository**

   ```bash
   git clone https://github.com/BrijeshRakhasiya/Intelligent-Document-Question-Answering-System-Using-Retrieval-Augmented-Generation-RAG.git
   cd Intelligent-Document-Question-Answering-System-Using-Retrieval-Augmented-Generation-RAG
   ```
   
2. **Create & activate a virtual environment**
         
         python3 -m venv .venv
        source .venv/bin/activate      # macOS / Linux 
        # .venv\Scripts\activate       # Windows
   
3. **Install dependencies**
          ```
            pip install -r requirements.txt
          ```

4. **Set required environment variables**

   You may need to set API keys or configuration variables.

# Running the System 
```

streamlit run final_app.py

```


# Configuration & Environment Variables

You may need to configure:

- API keys (GROQ, HuggingFace, etc.)

- Document folder / path for ingestion

- Vector store persistence directory

- Embedding model / parameters (chunk size, overlap)

- Retrieval method / similarity metric

- Chat / UI parameters (max tokens, temperature, prompt templates)

- DB configuration (SQLite path or MySQL connection string)

 **Consider using a .env file and python-dotenv to load environment variables.**

 # Database / Persistence

- SQLite (student.db): Used for demo / local deployments.

- MySQL (mysql_db.sql + sql.py): A SQL schema is provided for a MySQL backend if desired.

- Vector Store: Embedding store persisted to disk (e.g. via Chroma or similar).

**The sql.py module contains helper functions to insert metadata, logs, etc.**

# Extending & Customizing

- Add support for more document formats (Word, HTML, PPTX, etc.).

- Swap in different embedding models or vector DBs (e.g. FAISS, Pinecone, Weaviate).

- Add prompt templates, few-shot examples, or chain-of-thought reasoning.

- Build a more robust web UI with login, usage tracking, streaming responses.

- Add unit tests for ingestion, embedding, retrieval, answer generation.

- Add error handling for missing files, bad API responses, rate limits.

