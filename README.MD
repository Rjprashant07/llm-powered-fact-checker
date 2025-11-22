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

## 🏗️ System Architecture

The project implements a complete **Retrieval-Augmented Generation (RAG)** pipeline for multilingual fact-checking (Hindi + English).  
Below is the high-level architecture of the system:

```text
┌──────────────────────────────┐
│          User Input          │
│    (Hindi or English Text)   │
└───────────────┬──────────────┘
                │
                ▼
┌──────────────────────────────┐
│       Claim Extraction        │
│      (spaCy Multilingual)     │
└───────────────┬──────────────┘
                │ Extracted Claims
                ▼
┌──────────────────────────────┐
│   Semantic Vector Retrieval   │
│    (ChromaDB + Embeddings)    │
└───────────────┬──────────────┘
                │ Top-K Relevant Facts
                ▼
┌──────────────────────────────┐
│      LLM Reasoning &         │
│   Claim–Fact Comparison       │
│    (Mistral 7B Instruct)      │
└───────────────┬──────────────┘
                │ Verdict + Evidence
                ▼
┌──────────────────────────────┐
│        Final Output JSON      │
│  Verdict | Reasoning | Evidence |
└──────────────────────────────┘



### 🔍 Detailed Architecture Components

#### **1️⃣ Claim Extraction**
- Uses **spaCy `xx_sent_ud_sm`** multilingual model  
- Splits input into sentences  
- Handles **Hindi + English**  
- Each sentence becomes a claim  

#### **2️⃣ Trusted Fact Base**
- Facts stored in `facts.txt`  
- Facts may span multiple lines  
- Each fact separated by a blank line  
- Assigned unique IDs (`fact_001`, `fact_002`, …)  

#### **3️⃣ Embeddings & Vector Database**
- Uses **sentence-transformers (multilingual)** to embed facts  
- Stores embeddings in **ChromaDB**  
- Retrieves **Top-K** most semantically similar facts per claim  

#### **4️⃣ LLM Reasoning**
- **Mistral 7B Instruct (4-bit)** performs deeper analysis  
- Compares claim vs. retrieved facts  
- Outputs structured JSON with:  
  - `verdict`  
  - `evidence` (fact IDs)  
  - `reasoning`  

#### **5️⃣ Structured Final Output**
- Evidence IDs mapped back to fact statements  
- System generates final JSON:

```json
{
  "verdict": "Likely True",
  "reasoning": "The retrieved facts support the claim.",
  "evidence": ["fact_003"],
  "evidence_texts": ["..."]
}


## ⚙️ Installation (Local)
1. Clone the repository

git clone https://github.com/Rjprashant07/llm-powered-fact-checker.git
cd llm-powered-fact-checker


pip install -r requirements.txt
python -m spacy download xx_sent_ud_sm

