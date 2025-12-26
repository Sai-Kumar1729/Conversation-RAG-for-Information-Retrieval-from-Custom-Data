# Conversational RAG for Disaster Report Analysis

## Overview
This project implements a **Conversational Retrieval-Augmented Generation (RAG) system** to answer natural-language questions from an official disaster report. The application is built using **Streamlit + LangChain**, enabling interactive, context-aware Q&A over the **2023 Sikkim Flash Flood NDMA report**.

The system combines document chunking, semantic embeddings, vector search, and a conversational LLM to deliver grounded, report-based responses.

---

## Key Features
- PDF-based knowledge ingestion
- Semantic search using FAISS vector store
- Context-aware conversational memory
- Google Gemini-powered response generation
- Lightweight, interactive Streamlit UI

---

Tech Stack
----------

*   **Python**
    
*   **Streamlit**
    
*   **LangChain**
    
*   **FAISS**
    
*   **SentenceTransformers**
    
*   **Google Generative AI (Gemini)**

Setup & Runpip install -r requirements.txt

streamlit run app.py

Create a .env file:

GOOGLE\_API\_KEY=your\_api\_key\_here

Use Case
--------

*   Disaster response analysis
    
*   Government or NGO report Q&A
    
*   Policy review and situational awareness
    
*   Rapid information retrieval from static PDFs


## Architecture (High Level)
```text
PDF Report
   ↓
Document Loader (PyPDFLoader)
   ↓
Text Chunking (RecursiveCharacterTextSplitter)
   ↓
Embeddings (all-MiniLM-L6-v2)
   ↓
FAISS Vector Store
   ↓
Conversational Retrieval Chain
   ↓
Google Gemini LLM
   ↓
Streamlit Chat Interface





