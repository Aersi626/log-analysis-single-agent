# 📄 LLM-Powered Log Analysis

> Analyze Apache Camel integration logs using a Large Language Model + RAG architecture with Elasticsearch and Streamlit.

---

## 🚀 Overview

This project provides an interactive tool for analyzing integration logs and suggesting actionable fixes using a Large Language Model (LLM) combined with **retrieval-augmented generation (RAG)**.

It uses:
- 📄 **Elasticsearch**: To store and retrieve log data & vector embeddings.
- 🤖 **Ollama (LLM)**: To generate explanations, root causes, and fixes.
- 🔎 **HuggingFace Embeddings**: To vectorize log messages for retrieval.
- 🌐 **Streamlit**: To deliver an intuitive web-based interface.

---

## 📋 Features

✅ Log ingestion with metadata & embeddings  
✅ Vector search over logs using Elasticsearch  
✅ RAG-based analysis of logs: detect errors, explain root causes, propose fixes  
✅ Streamlit UI for querying & visualizing results  
✅ Health checks for dependencies (Elasticsearch & embedding service)  
✅ Configurable via environment variables  

## 🗂️ Project Structure

├── main.py # Streamlit app entry point
├── README.md # This file
├── requirements.txt # Python dependencies