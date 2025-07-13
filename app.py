import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load CSV file (make sure it's in the same directory)
df = pd.read_csv("Training Dataset.csv")
df = df.fillna("NA")  # Fill missing values

# Convert each row to a document string
docs = df.astype(str).apply(lambda row: " | ".join(row), axis=1).tolist()

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(docs, convert_to_tensor=False)

# Create FAISS index
dimension = doc_embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# Load lightweight generative model
generator = pipeline("text2text-generation", model="google/flan-t5-base")

# RAG chatbot function
def rag_chatbot(query, top_k=3):
    query_embedding = embedder.encode([query])[0]
    D, I = index.search(np.array([query_embedding]), top_k)
    context = "\n".join([docs[i] for i in I[0]])
    
    prompt = f"Answer the question based on the context below:\n\nContext:\n{context}\n\nQuestion: {query}"
    response = generator(prompt, max_length=150, do_sample=False)[0]['generated_text']
    
    return response

# CLI chatbot
if __name__ == "__main__":
    print("💬 Loan Q&A Chatbot (type 'exit' to quit)\n")
    while True:
        query = input("You: ")
        if query.lower() in ['exit', 'quit']:
            break
        answer = rag_chatbot(query)
        print("Bot:", answer, "\n")
