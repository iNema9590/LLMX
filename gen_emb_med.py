import pandas as pd
import re
import pickle
import requests
import numpy as np
from tqdm import tqdm
import faiss
from joblib import Parallel, delayed
from sentence_transformers import SentenceTransformer

tqdm.pandas()

# Load dataset
splits = {'train': 'question-answer-passages/train-00000-of-00001.parquet', 'test': 'question-answer-passages/test-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/enelpol/rag-mini-bioasq/" + splits["train"])
df1 = pd.read_parquet("hf://datasets/enelpol/rag-mini-bioasq/text-corpus/test-00000-of-00001.parquet")

# Clean text
df1['passage'] = df1['passage'].str.replace(r'[\n]', ' ', regex=True)
df['question'] = df['question'].str.replace(r'[\n]', ' ', regex=True)

# Embedding function
def gen_embs(qtext, model="nomic"):
    if model == "nomic":
        data = {"model": "nomic-embed-text", "prompt": qtext}
        response = requests.post('http://localhost:11434/api/embeddings', json=data)
        return np.array(response.json().get('embedding', []))  
    else:
        model = SentenceTransformer("abhinand/MedEmbed-large-v0.1")
        return model.encode(qtext, convert_to_numpy=True)

# Parallel embedding generation with joblib
num_jobs = 16  # Adjust based on available CPU cores
df1['embedding'] = Parallel(n_jobs=num_jobs, backend="loky")(delayed(gen_embs)(x, model='medemb') for x in tqdm(df1['passage'].tolist()))

# Save embeddings
with open('data/embed_bioasq_medemb.pkl', 'wb') as f:
    pickle.dump(df1['embedding'].tolist(), f)
print("Embeddings saved to embed_bioasq_medemb.pkl")

# Load and normalize embeddings
with open("data/embed_bioasq_medemb.pkl", "rb") as f:
    embeddings = np.array(pickle.load(f))

def normalize_embeddings(embeddings):
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

embeddings = normalize_embeddings(embeddings)

# FAISS Indexing with batching
def index_documents(method="faiss", index_name="medemb_bioasq_faiss"):
    if method == "faiss":
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        
        batch_size = 10000  # Adjust for memory efficiency
        for i in range(0, len(embeddings), batch_size):
            index.add(embeddings[i:i + batch_size])
        
        faiss.write_index(index, "data/medemb_bioasq_faiss.index")
        print("FAISS index saved.")
        return index

index_documents(method="faiss")
