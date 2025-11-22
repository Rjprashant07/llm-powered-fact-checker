import os
import json
from typing import List, Dict

import pandas as pd
import spacy
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import shutil
# Step 1: Load and structure your facts from a TXT file

facts = []


# Each fact can span multiple lines, and facts are separated by a blank line.
with open("facts.txt", "r", encoding="utf-8") as f:
    content = f.read()

# Split on blank lines (two or more newlines) → each block = one fact
raw_blocks = [blk.strip() for blk in content.split("\n\n") if blk.strip()]

for i, block in enumerate(raw_blocks):
    # Join the lines inside each block into one single string
    fact_text = " ".join(
        line.strip() for line in block.splitlines() if line.strip()
    )

    facts.append({
        "id": f"fact_{i+1:03d}",   # auto ID: fact_001, fact_002, ...
        "source": "txt_file",      # you can change this later if needed
        "statement": fact_text
    })

print(f"Total facts loaded: {len(facts)}")
for f in facts[:5]:   # show first 5 as a sanity check
    print(f["id"], "→", f["statement"])

# Directory on disk where Chroma will store the index
PERSIST_DIR = "fact_chroma_db"

#  Create embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

#  Prepare data for Chroma
texts = [f["statement"] for f in facts]  # the actual fact sentences/paragraphs
metadatas = [{"id": f["id"], "source": f["source"]} for f in facts]
ids = [f["id"] for f in facts]

#  Create / load Chroma vector store
# Remove existing persistent directory to ensure dimension consistency
shutil.rmtree(PERSIST_DIR, ignore_errors=True)
vectorstore = Chroma(
    collection_name="gov_facts",
    embedding_function=embedding_model,
    persist_directory=PERSIST_DIR
)


#  Add your facts to the vector store
vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
vectorstore.persist()

#  Create a retriever for later use
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

print("Vector store ready. Total facts embedded:", len(facts))    

# Load English model 
nlp = spacy.load("xx_sent_ud_sm")

def extract_claims(text: str) -> List[Dict]:
    doc = nlp(text)
    claims = []

    for sent in doc.sents:
        sent_text = sent.text.strip()
        if len(sent_text) < 5:
            continue  # skip very short bits

        claims.append({
            "claim_text": sent_text,
            "entities": []  # we skip entities for now, Hindi NER is not covered here
        })

    return claims

# Load LLM model and tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


model_id = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

gen_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id
)

def call_llm(prompt: str) -> str:
    out = gen_pipe(prompt, max_new_tokens=512, do_sample=False)[0]["generated_text"]
    completion = out[len(prompt):].strip()
    return completion

#  Retrieval function

def retrieve_facts_for_claim(claim_text: str, k: int = 5):
    docs = retriever.get_relevant_documents(claim_text)
    results = []
    for d in docs:
        results.append({
            "id": d.metadata.get("id"),
            "source": d.metadata.get("source"),
            "statement": d.page_content
        })
    return results

import json
import re

def classify_claim_with_evidence(claim_text: str, retrieved_facts):
    if not retrieved_facts:
        return {
            "verdict": "Unverifiable",
            "evidence": [],
            "reasoning": "No relevant facts were retrieved from the trusted fact base."
        }

    facts_str = ""
    for f in retrieved_facts:
        facts_str += f"- {f['statement']} (id: {f['id']})\n"

    prompt = f"""
You are a STRICT fact-checking assistant.

TASK:
Compare the CLAIM only with the VERIFIED FACTS provided.
Decide if the claim is supported, contradicted, or not decidable.

Rules:
- If facts clearly SUPPORT the claim → verdict = "Likely True"
- If facts clearly CONTRADICT the claim → verdict = "Likely False"
- If facts are related but do NOT clearly support or contradict → verdict = "Unverifiable"

Return ONLY a JSON object in this exact format:

{{
  "verdict": "Likely True | Likely False | Unverifiable",
  "evidence": ["id1", "id2"],
  "reasoning": "short explanation"
}}

CLAIM:
{claim_text}

VERIFIED FACTS:
{facts_str}
"""

    raw = call_llm(prompt)
    # For debugging, you can temporarily print:
    # print("RAW MODEL OUTPUT:\n", raw)

    # Try direct JSON load
    try:
        data = json.loads(raw)
        return data
    except:
        pass

    # Try to extract a {...} block from the text
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return data
    except Exception as e:
        print("JSON parse error:", e)
        print("RAW OUTPUT (first 300 chars):", raw[:300])

    # Final fallback
    return {
        "verdict": "Unverifiable",
        "evidence": [],
        "reasoning": f"Could not parse JSON from model output. Raw: {raw[:200]}"
    }


def check_text(text: str):

    results = []
    claims = extract_claims(text)

    emoji_map = {
        "Likely True": "✅ True",
        "Likely False": "❌ False",
        "Unverifiable": "🤷‍♂️ Unverifiable"
    }

    for c in claims:
        claim_text = c["claim_text"]

        retrieved = retrieve_facts_for_claim(claim_text)
        verdict_data = classify_claim_with_evidence(claim_text, retrieved)

        # map evidence texts
        evidence_texts = []
        for ev in verdict_data.get("evidence", []):
            match = next((f for f in retrieved if f["id"] == ev), None)
            if match:
                evidence_texts.append(match["statement"])

        results.append({
            "claim": claim_text,
            "entities": c["entities"],
            "verdict": verdict_data["verdict"],
            "emoji_verdict": emoji_map.get(verdict_data["verdict"], "🤷‍♂️ Unverifiable"),
            "evidence_ids": verdict_data["evidence"],
            "evidence": evidence_texts,
            "reasoning": verdict_data["reasoning"]
        })

    return {"input_text": text, "results": results}
