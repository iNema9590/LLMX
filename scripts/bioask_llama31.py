import sys
import os
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from SHapRAG import*
import pandas as pd
import pickle
import requests
import numpy as np
from tqdm import tqdm
import faiss
from scipy.stats import spearmanr, pearsonr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sentence_transformers import SentenceTransformer

splits = {'train': 'question-answer-passages/train-00000-of-00001.parquet', 'test': 'question-answer-passages/test-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/enelpol/rag-mini-bioasq/" + splits["train"])

df1 = pd.read_parquet("hf://datasets/enelpol/rag-mini-bioasq/text-corpus/test-00000-of-00001.parquet")
df1['passage']=df1['passage'].str.replace(r'[\n]', ' ', regex=True)
df['question']=df['question'].str.replace(r'[\n]', ' ', regex=True)

def gen_embs(qtext, model="nomic"):
    if model=="nomic":
    
        data = {
            "model": "nomic-embed-text",
            "prompt": qtext
        }
        return np.array(requests.post('http://localhost:11434/api/embeddings', json=data).json()['embedding'])
    else:
        return SentenceTransformer("abhinand/MedEmbed-large-v0.1").encode([qtext], convert_to_numpy=True)
    
with open("/data/embed_bioasq_medemb.pkl", "rb") as f:
    embeddings = pickle.load(f)

def normalize_embeddings(embeddings):
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

embeddings = normalize_embeddings(embeddings)

def index_documents(method="faiss", index_name="recipes_nomic", es_host="http://localhost:9200"):
    if method == "faiss":
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        faiss.write_index(index, "/data/bioasq_nomic_faiss.index")
        print("FAISS index saved.")
        return index
    elif method == "elasticsearch":
        es = Elasticsearch(es_host)
        mapping = {"mappings": {"properties": {"text": {"type": "text"}, "vector": {"type": "dense_vector", "dims": embeddings.shape[1]}}}}
        es.indices.create(index=index_name, body=mapping, ignore=400)
        for i, (text, vector) in enumerate(zip(documents, embeddings)):
            es.index(index=index_name, id=i, body={"text": text, "vector": vector.tolist()})
        print("Elasticsearch index created.")
        return es
# index_documents(method="faiss", index_name="bioasq_nomic_faiss", es_host="http://localhost:9200")
faiss_index = faiss.read_index("/data/medemb_bioasq_faiss.index")

def retrieve_documents(query, k=5):
    query_embedding = gen_embs(query, model="medemb")
    query_embedding = normalize_embeddings(query_embedding.reshape(1, -1))
    scores, indices = faiss_index.search(query_embedding, k)
    return [df1['passage'][i] for i in indices[0]], scores

num_questions_to_run = 30
print(f"Running experiments for {num_questions_to_run} questions...")

# Parameters for attribution methods
NUM_RETRIEVED_DOCS =9 
SEED = 42
# LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # Use a small, fast model for this demo
model_path = "meta-llama/Llama-3.2-1B-Instruct"  # Or your desired model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    # device_map="auto" could also be used if you prefer Transformers to handle initial placement,
    # but accelerator.prepare() will still manage the final device assignment per process.
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
# Store metrics for each question and method
all_metrics_data = []

for i in tqdm(range(num_questions_to_run), desc="Processing Questions"):
    query = df.question[i]
    print(f"\n--- Question {i+1}/{num_questions_to_run}: {query} ---")

    # 2. RAG Pipeline: Retrieve Documents

    docs, scores = retrieve_documents(query=query, k=NUM_RETRIEVED_DOCS)
    if not docs or len(docs) < NUM_RETRIEVED_DOCS:
        print(f"  Skipping query due to insufficient documents retrieved ({len(docs)}).")
        continue


    # 3. Instantiate ShapleyExperimentHarness
    #    This will pre-compute all utilities (2^N LLM calls)
    print(f"  Instantiating ShapleyExperimentHarness for n={len(docs)} docs...")
    harness = ShapleyExperimentHarness(
            items=docs,
            query=query,
            model=model,         # Pass the loaded model
            tokenizer=tokenizer, # Pass the loaded tokenizer
            verbose=True
        )
            
        # 4. Compute Attributions using various methods
    results_for_query = {}
    current_accelerator = harness.accelerator
    # Exact Shapley (Ground Truth)
    if current_accelerator.is_main_process:
        print(f"    Computing Exact Shapley...")
        results_for_query["Exact"] = harness.compute_exact_shap()


        # Approximate Methods
        m_samples = 64
        T_iterations = 100 # Fewer iterations for faster demo loop
        
        print(f"    Computing ContextCite (m={m_samples})...")
        results_for_query["ContextCite64"] = harness.compute_contextcite_weights(num_samples=m_samples, lasso_alpha=0.0, seed=SEED)

        print(f"    Computing ContextCite (m={32})...")
        results_for_query["ContextCite32"] = harness.compute_contextcite_weights(num_samples=m_samples, lasso_alpha=0.0, seed=SEED)
        
        print(f"    Computing WSS (m={32})...")
        results_for_query["WSS32"] = harness.compute_wss(num_samples=32, lasso_alpha=0.0, seed=SEED)

        print(f"    Computing WSS (m={m_samples})...")
        results_for_query["WSS64"] = harness.compute_wss(num_samples=m_samples, lasso_alpha=0.0, seed=SEED)
        
        print(f"    Computing WSS (m={100})...")
        results_for_query["WSS100"] = harness.compute_wss(num_samples=100, lasso_alpha=0.0, seed=SEED)


        print(f"    Computing TMC (T={64})...")
        results_for_query["TMC64"] = harness.compute_tmc_shap(num_iterations=T_iterations, performance_tolerance=0.001, seed=SEED)
        print(f"    Computing TMC (T={T_iterations})...")
        results_for_query["TMC"] = harness.compute_tmc_shap(num_iterations=T_iterations, performance_tolerance=0.001, seed=SEED)
        
        if beta_dist:
            print(f"    Computing BetaShap (U-shaped, T={T_iterations})...")
            results_for_query["BetaShap (U)"] = harness.compute_beta_shap(num_iterations=T_iterations, beta_a=0.5, beta_b=0.5, seed=SEED)

        print(f"    Computing LOO...")
        results_for_query["LOO"] = harness.compute_loo()

        # 5. Calculate and Store Metrics for this query
        exact_scores = results_for_query.get("Exact")
        if exact_scores is not None:
            for method, approx_scores in results_for_query.items():
                if method != "Exact" and approx_scores is not None:
                    if len(approx_scores) == len(exact_scores):
                        # Handle potential constant arrays for correlation
                        if np.all(exact_scores == exact_scores[0]) or np.all(approx_scores == approx_scores[0]):
                            pearson_c = 1.0 if np.allclose(exact_scores, approx_scores) else 0.0
                            spearman_c = 1.0 if np.allclose(exact_scores, approx_scores) else 0.0
                        else:
                          
                            pearson_c, _ = pearsonr(exact_scores, approx_scores)
                            spearman_c, _ = spearmanr(exact_scores, approx_scores)
             
                        
                        all_metrics_data.append({
                            "Question_Index": i,
                            "Query": query,
                            "Method": method,
                            "Pearson": pearson_c,
                            "Spearman": spearman_c
                        })
                    else:
                            print(f"    Score length mismatch for method {method} (Exact: {len(exact_scores)}, Approx: {len(approx_scores)}). Skipping metrics.")
        else:
            print(f"    Skipping metric calculation for Q{i} as Exact Shapley was not computed.")
    
    # Optional: Brief sleep to avoid overwhelming resources if LLM is remote
    # time.sleep(0.1)


# 6. Aggregate and Report Average Metrics
if all_metrics_data:
    metrics_df_all_questions = pd.DataFrame(all_metrics_data)
    
    print("\n\n--- Average Correlation Metrics Across All Questions ---")
    # Group by method and calculate mean, std for Pearson and Spearman
    average_metrics = metrics_df_all_questions.groupby("Method").agg(
        Avg_Pearson=("Pearson", "mean"),
        Avg_Spearman=("Spearman", "mean"),
        Num_Valid_Queries=("Query", "nunique") # Count how many queries had valid metrics for this method
    ).sort_values(by="Avg_Spearman", ascending=False)
    
    print(average_metrics.round(4))

    # You can also save metrics_df_all_questions to a CSV for detailed analysis
    metrics_df_all_questions.to_csv("../Experiment_data/shapley_rag_experiment_details.csv", index=False)
    average_metrics.to_csv("../Experiment_data/shapley_rag_experiment_summary.csv")
    current_accelerator.wait_for_everyone()
else:
    print("\nNo metrics were collected. This might be due to Exact Shapley not being run or errors.")
    current_accelerator.wait_for_everyone()

if current_accelerator.is_main_process:
    print("All processes finished.")

# import torch.distributed as dist

# if dist.is_initialized():
#     dist.destroy_process_group()