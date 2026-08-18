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
    # if len(chunks)<chunksize:
    #     print(chunks )

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

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)
print(index.ntotal)
query=input()
querys=model.encode([query])
D, I = index.search(querys, k=3)
# print(I,D)
retrieved_chunks = [chunks[i] for i in I[0]]
# print(retrieved_chunks)
sentences=[]
for chunk in retrieved_chunks:
    sentence=" ".join(chunk)
    sentences.append(sentence)
context="\n".join(sentences)
print(context)

prompt = f"""
Answer the question using ONLY the context below.
Be concise and do not make up information.

Context:
{context}

Question:
{query}
"""
from google import genai
import os
client = genai.Client(api_key=os.getenv("API_KEY"))

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt
)

print(response.text)