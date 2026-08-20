# Retrieval-Augmented Generation (RAG) System with FAISS and Gemini

## Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline that combines vector similarity search with a Large Language Model (LLM) to generate context-aware answers.

Instead of relying only on a language model, this system retrieves relevant information from a dataset and uses it to produce grounded responses.

---

## System Architecture

User Query  
→ Query Embedding  
→ FAISS Vector Search  
→ Top-K Relevant Chunks  
→ Context Construction  
→ Prompt Creation  
→ LLM (Gemini)  
→ Final Answer  

---

## Pipeline Explanation

### 1. Document Processing
Input documents are split into smaller chunks. Overlapping chunks are used to preserve context across boundaries.

### 2. Embedding Generation
Text chunks are converted into vector representations using SentenceTransformers (all-MiniLM-L6-v2).

### 3. Vector Storage (FAISS)
Embeddings are stored in FAISS for efficient similarity search using L2 distance.

### 4. Retrieval
The user query is converted into an embedding, and FAISS retrieves the top-k most relevant chunks.

### 5. Context Construction
Retrieved chunks are combined into a single context block.

### 6. Response Generation
The context and query are passed to a Large Language Model (Google Gemini), which generates the final answer.

---

## Example

Query:
What is Artificial Intelligence?

Output:
Artificial Intelligence (AI) is the capability of computational systems to perform tasks such as learning, reasoning, problem-solving, and decision-making.

---

## Tech Stack

- Python  
- SentenceTransformers  
- FAISS  
- Google Gemini API  
- NumPy  

---

## Installation

pip install sentence-transformers faiss-cpu google-generativeai numpy

---

## Key Concepts

- Semantic similarity vs keyword matching  
- Embeddings and vector search  
- FAISS indexing and retrieval  
- Difference between retrieval and generation  
- Role of context in LLM-based systems  

---

## Limitations

- Small dataset limits knowledge coverage  
- Basic chunking strategy  
- No reranking of retrieved results  
- Possible redundancy in context  

---

## Future Improvements

- Add reranking for better retrieval accuracy  ##done (added reranking)
- Improve chunking and preprocessing   ##done
- Add context filtering and deduplication   ##done (added mmr)
- Build a user interface  
- Introduce caching for performance  

---

## Project Type

Educational project demonstrating the core concepts behind RAG systems.
