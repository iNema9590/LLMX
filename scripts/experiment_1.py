import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

import argparse
import gc
import random
import string
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, pearsonr, rankdata, spearmanr
from sklearn.metrics import ndcg_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import hit_rate_at_k, precision_at_k, reciprocal_rank

from SHapRAG import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, required=True)
parser.add_argument("--precision", default='bf16', type = str)
parser.add_argument("--model_name", type = str, required = True)
parser.add_argument("--shuffle", action="store_true")
args = parser.parse_args()

available_models = {"qwen_3B" : "Qwen/Qwen2.5-3B-Instruct", 
                    "llama_3B" : "meta-llama/Llama-3.2-3B-Instruct", 
                    "llama_8B" : "meta-llama/Llama-3.1-8B-Instruct"}

start = time.time()

# Construct full path to CSV file in same folder

csv_path = os.path.join(current_dir, f'../data/{args.dataset}.csv')

df = pd.read_csv(csv_path)
df_save_results = pd.DataFrame(columns = ["query", "context", "provided_answer", "scoring", "doc_id"])
print("Data Loaded!")

num_questions_to_run = 10
print(f"Running experiments for {num_questions_to_run} questions...")

# Parameters
NUM_RETRIEVED_DOCS = 10
SEED = 42
DATASET_NAME = f"{args.dataset}"
PRECISION = f"{args.precision}"
SHUFFLE = ["SHUFFLE" if args.shuffle else "VANILLA"]
random.seed(42) # SEED SO THAT WE CAN REPLICATE IF NEEDED

if SHUFFLE: 
    index_list = list(range(10)) # context size

# Initialize Accelerator ONCE
accelerator_main = Accelerator(mixed_precision=PRECISION)

if accelerator_main.is_main_process:
    print(f"Main Script: Loading model...")
model_path = available_models[args.model_name]
MODEL_NAME = model_path.split('/')[1]

model_cpu = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
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
    query = df.question.loc[i]
    

    if accelerator_main.is_main_process:
        print(f"\n--- Question {i+1}/{num_questions_to_run}: {query[:60]}... ---")
    
    if isinstance(df.context, list) == True: 
        docs= df['context'].loc[i]
        flags = df["doc_id"].loc[i]
    else: 
        docs= eval(df['context'].loc[i])
        flags = eval(df["doc_id"].loc[i])

    if SHUFFLE == True: 
        index_list = list(range(10))
        random.seed(i) # different seed for each query/line
        random.shuffle(index_list)
        docs = [docs[j] for j in index_list]
        flags = [flags[j] for j in index_list]

    utility_cache_base_dir = f"Experiment_data/{DATASET_NAME}/{MODEL_NAME}/utilities_cache3bcp"
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

    if accelerator_main.is_main_process:
        results_for_query["Exact"] = harness.compute_exact_shap()

        m_samples_map = {"S": 32, "M": 64, "L": 100} # Example sample sizes
        T_iterations_map = {"S": 32, "M": 64, "L":100}  # Example iterations

        for size_key, num_s in m_samples_map.items():
            if 2**len(docs) < num_s and size_key != "L": # Avoid sampling more than possible, except for a default
                actual_samples = max(1, 2**len(docs)-1 if 2**len(docs)>0 else 1) # Ensure at least 1 sample if possible
            else:
                actual_samples = num_s

            if actual_samples > 0: 
                results_for_query[f"ContextCite{actual_samples}"] = harness.compute_contextcite_weights(num_samples=actual_samples, sampling="uniform",  seed=SEED) #lasso_alpha=0.01,
                results_for_query[f"WSS{actual_samples}"] = harness.compute_wss(num_samples=actual_samples,  seed=SEED, sampling="uniform") #lasso_alpha=0.01,
                results_for_query[f"BetaShap (U){actual_samples}"] = harness.compute_beta_shap(num_iterations_max=T_iterations_map[size_key], beta_a=0.5, beta_b=0.5, max_unique_lookups=actual_samples, seed=SEED)
                results_for_query[f"TMC{actual_samples}"] = harness.compute_tmc_shap(num_iterations_max=T_iterations_map[size_key], performance_tolerance=0.001, max_unique_lookups=actual_samples, seed=SEED)
        
        results_for_query["LOO"] = harness.compute_loo()

        exact_scores = results_for_query.get("Exact")

        df_save_results.loc[i, "query"] = query
        df_save_results.loc[i, "scoring"] = [results_for_query]
        # print("Flags Value: ", flags)
        df_save_results.loc[i, "doc_id"] = [[flags]]
        df_save_results.loc[i, "context"] = [[docs]]
        df_save_results.loc[i, "provided_answer"] = harness.target_response


        print("RESULTS FOR QUERY: ", df_save_results)

        if exact_scores is not None: # proceed if the ground truth scores are avaialble for this query
            positive_exact_score = np.clip(exact_scores, a_min=0.0, a_max=None) # FOR NDGC SCORE COMPUTATION
            for method, approx_scores in results_for_query.items(): # Iterate over each method and its attributed scores
                
                if method != "Exact" and approx_scores is not None:
                    if len(approx_scores) == len(exact_scores):
                        if np.all(exact_scores == exact_scores[0]) or np.all(approx_scores == approx_scores[0]): # check if all elements in exact_scores / approx_scores are the same [1,1,1,1,1] => correlation becomes undefined in that case
                            # spearman and pearson correlations rely on variance, so if the values are constant (no variance) we get an error or no result
                            # if both vectors are consta    nt but they are almost the same we have correlation =1, if they are different (one is [1,1,1] the other is [2,2,2]) we set correlation to 0.0
                            pearson_c = 1.0 if np.allclose(exact_scores, approx_scores) else 0.0 
                            spearman_c = 1.0 if np.allclose(exact_scores, approx_scores) else 0.0
                            kendall_c = 1.0 if np.allclose(exact_scores, approx_scores) else 0.0
                            precision_k = 1.0 if np.allclose(exact_scores, approx_scores) else 0.0
                            hit_rate_k = 1.0 if np.allclose(exact_scores, approx_scores) else 0.0
                            RR = 1.0 if np.allclose(exact_scores, approx_scores) else 0.0
                        else:
                            pearson_c, _ = pearsonr(exact_scores, approx_scores) #pearson measures linear correlation
                            spearman_c, _ = spearmanr(exact_scores, approx_scores) # spearman measures rank correlation (how well the order matches)

                            exact_ranks = rankdata(-np.array(exact_scores), method="average") # rank scores with the smallest =1 and when there is a tie assign the average rank
                            approx_ranks = rankdata(-np.array(approx_scores), method = "average")
                            kendall_c, _ = kendalltau(exact_ranks, approx_ranks) # return tau and pval (if pval is < 0.005 we can say that correlation is statistically significant) 
                            precision_k = precision_at_k(exact_scores, approx_scores) # k = 3 by default
                            hit_rate_k = hit_rate_at_k(exact_scores, approx_scores)
                            RR = reciprocal_rank(exact_scores, approx_scores)
                        
                        ndgc_scoring  = ndcg_score(
                            [positive_exact_score], 
                            [approx_scores],
                            k = 3 # focus on top k document scoring
                        )
                        
                        all_metrics_data.append({
                            "Question_Index": i, "Query": query, "Method": method,
                            "Pearson": pearson_c, "Spearman": spearman_c, "NDCG" : ndgc_scoring, "KendallTau" : kendall_c,
                            "Precision_at_k" : precision_k, "Hit_rate_at_k" : hit_rate_k, "Reciprocal_Rank" : RR, 
                            "Num_Items": len(docs), 
                        })

                        # print("CHECK ALL RESULTS: ", all_metrics_data)
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

# SAVE EXACT SHAP RESULTS FOR ANALYSIS
df_save_results.to_csv(f"../Experiment_data/{DATASET_NAME}/{MODEL_NAME}/results_{SHUFFLE}.csv", index=False)

# Save
df_save_results.to_json(f"../Experiment_data/{DATASET_NAME}/{MODEL_NAME}/results_{SHUFFLE}.json", orient="records", lines=True)

# Aggregate and Report Average Metrics (Only on main process)
if accelerator_main.is_main_process:
    if all_metrics_data:
        metrics_df_all_questions = pd.DataFrame(all_metrics_data)
        
        print("\n\n--- Average Correlation Metrics Across All Questions ---")
        average_metrics = metrics_df_all_questions.groupby("Method").agg(
            Avg_Pearson=("Pearson", "mean"),
            Avg_Spearman=("Spearman", "mean"),
            Avg_Kendall =("KendallTau", "mean"),
            Avg_NDCG = ("NDCG", "mean"),
            Avg_Precision_at_k = ("Precision_at_k", "mean"), 
            Avg_Hit_rate_at_k = ("Hit_rate_at_k", "mean"), 
            Avg_Reciprocal_rank = ("Reciprocal_Rank", 'mean'), 
            Num_Valid_Queries=("Query", "nunique")
        ).sort_values(by="Avg_Spearman", ascending=False)
        
        print(average_metrics.round(4))

        details_path = f"Experiment_data/{DATASET_NAME}/{MODEL_NAME}/{SHUFFLE}_shapley_rag_experiment_details3bcp.csv"
        summary_path = f"Experiment_data/{DATASET_NAME}/{MODEL_NAME}/{SHUFFLE}_shapley_rag_experiment_summary3bcp.csv"
        os.makedirs(os.path.dirname(details_path), exist_ok=True)
        
        metrics_df_all_questions.to_csv(details_path, index=False)
        average_metrics.to_csv(summary_path)
    else:
        print("\nNo metrics were collected. This might be due to all calculations failing or only non-main processes running sections.")

# Final synchronization before script ends
accelerator_main.wait_for_everyone()
if accelerator_main.is_main_process:
    print("Script finished.")

if torch.distributed.is_available() and torch.distributed.is_initialized():
    if accelerator_main.is_local_main_process:
        print(f"Rank {accelerator_main.process_index} (Local Main): Manually destroying process group...")
    torch.distributed.destroy_process_group()
    if accelerator_main.is_local_main_process:
        print(f"Rank {accelerator_main.process_index} (Local Main): Process group destroyed.")
else:
    if accelerator_main.is_local_main_process:
        print(f"Rank {accelerator_main.process_index} (Local Main): Distributed environment not initialized or not available, skipping destroy_process_group.")

if accelerator_main.is_main_process:
    print("Script fully exited.")
end = time.time()
print(f"Execution time: {end - start:.4f} seconds")