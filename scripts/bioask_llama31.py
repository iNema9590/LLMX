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
import gc
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

num_questions_to_run = 30 # Keep small for testing
print(f"Running experiments for {num_questions_to_run} questions...")

# Parameters
NUM_RETRIEVED_DOCS = 8
SEED = 42

# Initialize Accelerator ONCE
accelerator_main = Accelerator(mixed_precision="bf16") 

if accelerator_main.is_main_process:
    print(f"Main Script: Loading model...")
model_path = "meta-llama/Llama-3.2-1B-Instruct" # Example, ensure this path is correct
model_cpu = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model_cpu.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model_cpu, 'generation_config') and model_cpu.generation_config is not None: # Check if generation_config exists
         model_cpu.generation_config.pad_token_id = tokenizer.pad_token_id

if accelerator_main.is_main_process:
    print(f"Main Script: Preparing model with Accelerator...")
prepared_model = accelerator_main.prepare(model_cpu)
unwrapped_prepared_model = accelerator_main.unwrap_model(prepared_model)
unwrapped_prepared_model.eval()
if accelerator_main.is_main_process:
    print(f"Main Script: Model prepared and set to eval.")

all_metrics_data = []

for i in tqdm(range(num_questions_to_run), desc="Processing Questions", disable=not accelerator_main.is_main_process):
    query = df.question[i][:10]
    if accelerator_main.is_main_process:
        print(f"\n--- Question {i+1}/{num_questions_to_run}: {query[:60]}... ---")

    docs, scores = retrieve_documents(query=query, k=NUM_RETRIEVED_DOCS)

    utility_cache_base_dir = "../Experiment_data/bioask_utilities_cache"
    utility_cache_filename = f"utilities_q_idx{i}_n{len(docs)}.pkl" # More robust naming
    current_utility_path = os.path.join(utility_cache_base_dir, utility_cache_filename)
    
    if accelerator_main.is_main_process: # Only main process creates directories
        os.makedirs(os.path.dirname(current_utility_path), exist_ok=True)
        print(f"  Instantiating ShapleyExperimentHarness for Q{i} (n={len(docs)} docs)...")
    
    # Synchronize before creating harness instance to ensure directory exists for all if utility_path is used by others
    accelerator_main.wait_for_everyone() 

    harness = ShapleyExperimentHarness(
        items=docs,
        query=query,
        prepared_model_for_harness=prepared_model,
        tokenizer_for_harness=tokenizer,
        accelerator_for_harness=accelerator_main,
        verbose=True, # Verbose can be True, but most prints inside harness are main_process guarded
        utility_path=current_utility_path
    )
            
    results_for_query = {}
    # Computation of attributions should only happen on the main process
    # as they modify `results_for_query` which is then used to populate `all_metrics_data`.
    # `all_metrics_data` is a simple list on the main process.
    if accelerator_main.is_main_process:
        results_for_query["Exact"] = harness.compute_exact_shap()

        m_samples_map = {"S": 32, "M": 64, "L": 100} # Example sample sizes
        T_iterations_map = {"S": 64, "M": 100}  # Example iterations

        for size_key, num_s in m_samples_map.items():
            if 2**len(docs) < num_s and size_key != "M": # Avoid sampling more than possible, except for a default
                actual_samples = max(1, 2**len(docs)-1 if 2**len(docs)>0 else 1) # Ensure at least 1 sample if possible
            else:
                actual_samples = num_s

            if actual_samples > 0: # Ensure positive number of samples
                results_for_query[f"ContextCite{actual_samples}"] = harness.compute_contextcite_weights(num_samples=actual_samples, lasso_alpha=0.0, seed=SEED)
                
                results_for_query[f"WSS{actual_samples}"] = harness.compute_wss(num_samples=actual_samples, lasso_alpha=0.0, seed=SEED)

        for size_key, num_t in T_iterations_map.items():
            results_for_query[f"TMC{num_t}"] = harness.compute_tmc_shap(num_iterations=num_t, performance_tolerance=0.001, seed=SEED)
        
        if beta_dist: # beta_dist needs to be defined or imported (e.g., from scipy.stats)
            results_for_query["BetaShap (U)"] = harness.compute_beta_shap(num_iterations=T_iterations_map['M'], beta_a=0.5, beta_b=0.5, seed=SEED)

        results_for_query["LOO"] = harness.compute_loo()

        exact_scores = results_for_query.get("Exact")
        if exact_scores is not None:
            for method, approx_scores in results_for_query.items():
                if method != "Exact" and approx_scores is not None:
                    if len(approx_scores) == len(exact_scores):
                        if np.all(exact_scores == exact_scores[0]) or np.all(approx_scores == approx_scores[0]):
                            pearson_c = 1.0 if np.allclose(exact_scores, approx_scores) else 0.0
                            spearman_c = 1.0 if np.allclose(exact_scores, approx_scores) else 0.0
                        else:
                            pearson_c, _ = pearsonr(exact_scores, approx_scores)
                            spearman_c, _ = spearmanr(exact_scores, approx_scores)
                        
                        all_metrics_data.append({
                            "Question_Index": i, "Query": query, "Method": method,
                            "Pearson": pearson_c, "Spearman": spearman_c, "Num_Items": len(docs)
                        })
                    else:
                        print(f"    Score length mismatch for method {method} (Exact: {len(exact_scores)}, Approx: {len(approx_scores)}). Skipping metrics.")
        else:
            print(f"    Skipping metric calculation for Q{i} as Exact Shapley was not computed or failed.")
    
    accelerator_main.wait_for_everyone() 
    del harness
   
    if torch.cuda.is_available():
        if accelerator_main.is_main_process: # Print from one process
            print(f"Attempting to empty CUDA cache on rank {accelerator_main.process_index} after Q{i}")
        torch.cuda.empty_cache()
        gc.collect()
        if accelerator_main.is_main_process:
            print(f"CUDA cache empty attempt complete on rank {accelerator_main.process_index}.")
    accelerator_main.wait_for_everyone()


# Aggregate and Report Average Metrics (Only on main process)
if accelerator_main.is_main_process:
    if all_metrics_data:
        metrics_df_all_questions = pd.DataFrame(all_metrics_data)
        
        print("\n\n--- Average Correlation Metrics Across All Questions ---")
        average_metrics = metrics_df_all_questions.groupby("Method").agg(
            Avg_Pearson=("Pearson", "mean"),
            Avg_Spearman=("Spearman", "mean"),
            Num_Valid_Queries=("Query", "nunique")
        ).sort_values(by="Avg_Spearman", ascending=False)
        
        print(average_metrics.round(4))

        details_path = "../Experiment_data/bioask_results/shapley_rag_experiment_details.csv"
        summary_path = "../Experiment_data/bioask_results/shapley_rag_experiment_summary.csv"
        os.makedirs(os.path.dirname(details_path), exist_ok=True)
        
        metrics_df_all_questions.to_csv(details_path, index=False)
        average_metrics.to_csv(summary_path)
    else:
        print("\nNo metrics were collected. This might be due to all calculations failing or only non-main processes running sections.")

# Final synchronization before script ends
accelerator_main.wait_for_everyone()
if accelerator_main.is_main_process:
    print("Script finished.")