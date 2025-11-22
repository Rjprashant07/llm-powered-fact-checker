# 🕵️‍♂️ Multilingual RAG-Based Fact Checker (Hindi + English)

A Retrieval-Augmented Generation (RAG) system that automatically verifies factual claims from news or social media content using NLP, vector search, and LLM reasoning.  
Supports both **Hindi and English**, perfect for fact-checking government announcements and public statements.

---

## 📑 Table of Contents

- [⭐ Features](#-features)
- [🔧 How It Works](#-how-it-works)
  - [1️⃣ Claim Extraction](#1️⃣-claim-extraction)
  - [2️⃣ Fact Base](#2️⃣-fact-base)
  - [3️⃣ Embeddings--Vector-Search](#3️⃣-embeddings--vector-search)
  - [4️⃣ LLM Reasoning](#4️⃣-llm-reasoning)
- [🏗️ Architecture](#️-architecture)
- [📁 Project Structure](#-project-structure)
- [⚙️ Installation (Local)](#️-installation-local)
- [💻 Running in Google Colab](#-running-in-google-colab)
- [🧠 Tech Stack](#-tech-stack)
- [📸 Screenshots](#-screenshots)
- [📜 License](#-license)
- [🙌 Contributing](#-contributing)

---

## ⭐ Features

- 🔍 **Automatic Claim Extraction** (spaCy multilingual)
- 📚 **Semantic Fact Retrieval** (ChromaDB + sentence-transformers)
- 🤖 **LLM-Based Comparison** (Mistral 7B Instruct, 4-bit)
- 📝 **Structured Verdicts**  
- 🌐 **Hindi + English Support**
- ⚡ Fast, lightweight, and local friendly

---

## 🔧 How It Works

The system follows a **5-step RAG pipeline** for automated fact-checking.

### 1️⃣ Claim Extraction
- Uses spaCy `xx_sent_ud_sm` for multilingual sentence segmentation  
- Each meaningful sentence becomes a claim

### 2️⃣ Fact Base
- Verified statements stored in `facts.txt`
- Each fact may span multiple lines
- Cleaned + assigned unique IDs  

### 3️⃣ Embeddings & Vector Search
- Facts embedded using multilingual sentence transformers  
- Stored inside **ChromaDB**
- For every claim, top-k similar facts are retrieved  

### 4️⃣ LLM Reasoning
Mistral 7B Instruct evaluates:
- the claim  
- the retrieved facts  

Returns:
- verdict  
- evidence IDs  
- reasoning  

  

---

## 🏗️ Architecture

User Input
     │
     ▼
Claim Extraction (spaCy)
     │
     ▼
Semantic Retrieval (Chroma + Multilingual Embeddings)
     │
     ▼
LLM Comparison (Mistral 7B Instruct)
     │
     ▼
Structured Output (verdict + reasoning + evidence)




## 📁 Project Structure

fact-checker/
│
├── pipeline.py          # Complete backend RAG logic
├── facts.txt            # Verified fact database
├── requirements.txt     # Dependencies
└── README.md




---

## 📸 Screenshots

Below are some screenshots showing how the fact-checker system works.

---

### 🖥️ Example Input and output
<img src="Screenshot 2025-11-22 203008.png" width="700">

---

### 📊 Example Input and output
<img src="Screenshot 2025-11-22 203059.png" width="700">

---

### 🧠 Example Input and output
<img src="Screenshot 2025-11-22 203346.png" width="700">

### 🧠 Example Input and output
<img src="Screenshot 2025-11-22 204133.png" width="700">

### 🧠 Example Input and output
<img src="Screenshot 2025-11-22 204347.png" width="700">

### 🧠 Example Input and output
<img src="Screenshot 2025-11-22 204544.png" width="700">
---

  






## ⚙️ Installation (Local)

### 1. Clone the repository
```bash
git clone https://github.com/Rjprashant07/llm-powered-fact-checker.git
cd llm-powered-fact-checker

pip install -r requirements.txt
python -m spacy download xx_sent_ud_sm

