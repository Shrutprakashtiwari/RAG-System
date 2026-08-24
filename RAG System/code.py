# docs = [
#     "Artificial intelligence (AI) is the capability of computational systems to perform tasks typically associated with human intelligence, such as learning, reasoning, problem-solving, perception, and decision-making. It is a field of research in engineering, mathematics, and computer science that develops and studies methods and software that enable machines to perceive their environment and use learning and intelligence to take actions that maximise their chances of achieving defined goals.High-profile applications of AI include advanced web search engines, chatbots, virtual assistants, autonomous vehicles, play and analysis in strategy games (e.g., chess and Go), and content generation (e.g. images, audio, and videos).The traditional goals of AI research include learning, reasoning, knowledge representation, planning, natural language processing, and perception, as well as support for robotics.[a] To reach these goals, AI researchers use techniques including state space search and mathematical optimisation, formal logic, artificial neural networks, and methods based on statistics, operations research, and economics.[b] AI also draws upon psychology, linguistics, philosophy, neuroscience, and other fields.[2] Some companies, such as OpenAI, Google DeepMind, and Meta, aim to create artificial general intelligence (AGI)—AI that can complete nearly any cognitive task at least as well as a human."
# ]
import fitz
import re

doc = fitz.open("/content/Azure RTOS_Whitepaper.pdf")
docs = ""

for page in doc:
    docs += page.get_text()

docs = re.sub(r'\.{2,}', '', docs)

from sentence_transformers import SentenceTransformer
import numpy as np

def chunk(docs, chunksize=500, overlap=100):
    words = docs.split()
    chunks = []

    start = 0
    while start < len(words):
        chun = words[start:start + chunksize]
        chunk_text = " ".join(chun)
        chunks.append(chunk_text)
        start += chunksize - overlap

    return chunks

def embedding(chunks, model):
    embeddd = model.encode(chunks)
    embeddd = np.array(embeddd).astype("float32")
    import faiss
    faiss.normalize_L2(embeddd)
    print(embeddd.shape)
    return embeddd

model = SentenceTransformer("all-MiniLM-L6-v2")
chunks = chunk(docs)
embeddings = embedding(chunks, model)

# BM25 setup
from rank_bm25 import BM25Okapi
tokenized_chunks = [chunk.split() for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)

import faiss

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)
print(index.ntotal)

query = input()
querys = model.encode([query]).astype("float32")
faiss.normalize_L2(querys)

D, I = index.search(querys, k=8)
print(I, D)

# HYBRID SEARCH START
tokenized_query = query.split()
bm25_scores = bm25.get_scores(tokenized_query)

faiss_scores = D[0]
faiss_indices = I[0]

combined = []

for idx, f_score in zip(faiss_indices, faiss_scores):
    b_score = bm25_scores[idx]
    score = 0.7 * f_score + 0.3 * b_score
    combined.append((idx, score))

combined = sorted(combined, key=lambda x: x[1], reverse=True)
new_indices = [idx for idx, _ in combined]

I[0] = new_indices
# HYBRID SEARCH END

retrieved_chunks = [chunks[i] for i in I[0]]
print(retrieved_chunks)

retrieved_embeddings = embeddings[I[0]]

sentences = []

def mmr(querys, retrieved_embeddings, top_k=3, lambda_param=0.7):
    import numpy as np

    query = querys / np.linalg.norm(querys, axis=1, keepdims=True)
    docs = retrieved_embeddings / np.linalg.norm(retrieved_embeddings, axis=1, keepdims=True)

    relevance = (query @ docs.T)[0]

    selected_indices = []

    for _ in range(top_k):
        best_score = -1e9
        best_idx = -1

        for i in range(len(docs)):
            if i in selected_indices:
                continue

            rel = relevance[i]

            if len(selected_indices) == 0:
                diversity = 0
            else:
                sim = [docs[i] @ docs[j] for j in selected_indices]
                diversity = max(sim)

            score = lambda_param * rel - (1 - lambda_param) * diversity

            if score > best_score:
                best_score = score
                best_idx = i

        selected_indices.append(best_idx)

    return selected_indices

selected = mmr(querys, retrieved_embeddings, top_k=3)

final_chunks = [retrieved_chunks[i] for i in selected]

sentences = final_chunks
context = "\n".join(sentences)

print(context)

prompt = f"""
Answer the question using ONLY the context below.
Be concise and do not make up information.

Context:
{context}

Question:
{query}
"""
import os
client = genai.Client(api_key=os.getenv("API_KEY"))

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt
)

print(response.text)
