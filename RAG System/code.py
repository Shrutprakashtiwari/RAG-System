docs = [
    "Artificial intelligence (AI) is the capability of computational systems to perform tasks typically associated with human intelligence, such as learning, reasoning, problem-solving, perception, and decision-making. It is a field of research in engineering, mathematics, and computer science that develops and studies methods and software that enable machines to perceive their environment and use learning and intelligence to take actions that maximise their chances of achieving defined goals.High-profile applications of AI include advanced web search engines, chatbots, virtual assistants, autonomous vehicles, play and analysis in strategy games (e.g., chess and Go), and content generation (e.g. images, audio, and videos).The traditional goals of AI research include learning, reasoning, knowledge representation, planning, natural language processing, and perception, as well as support for robotics.[a] To reach these goals, AI researchers use techniques including state space search and mathematical optimisation, formal logic, artificial neural networks, and methods based on statistics, operations research, and economics.[b] AI also draws upon psychology, linguistics, philosophy, neuroscience, and other fields.[2] Some companies, such as OpenAI, Google DeepMind, and Meta, aim to create artificial general intelligence (AGI)—AI that can complete nearly any cognitive task at least as well as a human."
]

from sentence_transformers import SentenceTransformer
import numpy as np

def chunk(docs,chunksize=100,overlap=50):
    chunks=[]
    for i in docs:
        i=i.split()
        start=0
        while start<len(i):
            chun=i[start:start+chunksize]
            chunks.append(chun)
            start=start+chunksize-overlap
    if len(chunks)<chunksize:
        print(chunks )
    return chunks


def embedding(chunks,model):
    chunks=[" ".join(chun)for chun in chunks]
    embeddd=model.encode(chunks)
    print(embeddd.shape)
    return embeddd

model = SentenceTransformer("all-MiniLM-L6-v2")
chunks=chunk(docs)
embeddings=embedding(chunks,model)

import faiss

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

print(index.ntotal)

query=input()
querys=model.encode([query])

D, I = index.search(querys, k=8)
print(I,D)

retrieved_chunks = [chunks[i] for i in I[0]]
print(retrieved_chunks)

retrieved_embeddings = embeddings[I[0]]

from sentence_transformers import CrossEncoder

cross_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

pairs = [(query, " ".join(chunk)) for chunk in retrieved_chunks]

scores = cross_model.predict(pairs)

sorted_idx = np.argsort(scores)[::-1]

top_n = 5
top_idx = sorted_idx[:top_n]

retrieved_chunks = [retrieved_chunks[i] for i in top_idx]
retrieved_embeddings = retrieved_embeddings[top_idx]

def mmr(querys, retrieved_embeddings, top_k=3, lambda_param=0.7):

    import numpy as np

    query = querys / np.linalg.norm(querys, axis=1, keepdims=True)
    chunks = retrieved_embeddings / np.linalg.norm(retrieved_embeddings, axis=1, keepdims=True)

    relevance = (query @ chunks.T)[0]

    selected_indices = []

    for _ in range(top_k):

        best_score = -1e9
        best_idx = -1

        for i in range(len(chunks)):

            if i in selected_indices:
                continue

            rel = relevance[i]

            if len(selected_indices) == 0:
                diversity = 0
            else:
                sim = [chunks[i] @ chunks[j] for j in selected_indices]
                diversity = max(sim)

            score = lambda_param * rel - (1 - lambda_param) * diversity

            if score > best_score:
                best_score = score
                best_idx = i

        selected_indices.append(best_idx)

    return selected_indices


selected = mmr(querys, retrieved_embeddings, top_k=3)

final_chunks = [retrieved_chunks[i] for i in selected]

sentences = [" ".join(chunk) for chunk in final_chunks]
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
