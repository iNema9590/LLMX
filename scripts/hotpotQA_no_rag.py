import sys
import os
# current_dir = os.getcwd()
# parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
# sys.path.append(parent_dir)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from SHapRAG import *
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr, rankdata, kendalltau
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc
import time
from sklearn.metrics import ndcg_score

start = time.time()
# splits = {'train': 'question-answer-passages/train-00000-of-00001.parquet', 'test': 'question-answer-passages/test-00000-of-00001.parquet'}
# df = pd.read_parquet("hf://datasets/enelpol/rag-mini-bioasq/" + splits["train"])

# Construct full path to CSV file in same folder
# csv_path = os.path.join(current_dir, 'hotpotQA_sample.csv')
csv_path = os.path.join(current_dir, 'hotpotQA_sample_flagged.csv')

df = pd.read_csv(csv_path)
df_exact_shap = pd.DataFrame(columns = ["query", "scores", "doc_flags"])
print("Data Loaded!")

# df1 = pd.read_parquet("hf://datasets/enelpol/rag-mini-bioasq/text-corpus/test-00000-of-00001.parquet")
# df1.set_index('id', inplace=True)

# df1['passage']=df1['passage'].str.replace(r'[\n]', ' ', regex=True)
# df['question']=df['question'].str.replace(r'[\n]', ' ', regex=True)
# df = df[df['relevant_passage_ids'].apply(len) >= 10].reset_index(drop=True)

num_questions_to_run = 3
print(f"Running experiments for {num_questions_to_run} questions...")

# Parameters
NUM_RETRIEVED_DOCS = 10
SEED = 42
DATASET_NAME = "hotpotQA"
PRECISION = "bf16"

# Initialize Accelerator ONCE
accelerator_main = Accelerator(mixed_precision=PRECISION)

if accelerator_main.is_main_process:
    print(f"Main Script: Loading model...")
model_path = "Qwen/Qwen2.5-3B-Instruct" # Example, ensure this path is correct
# model_path = "meta-llama/Llama-3.2-3B-Instruct" # Example, ensure this path is correct
# model_path = "meta-llama/Llama-3.1-8B-Instruct" # Example, ensure this path is correct
MODEL_NAME = model_path.split('/')[0]

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
    
    docs= eval(df['context'].loc[i])
    flags = eval(df["documents_flag"].loc[i])

    # experiment with copies
    # numbers = random.sample(range(1, len(docs)), 3)
    # docs[numbers[0]]=docs[numbers[1]]
    # docs[numbers[2]]=docs[numbers[1]]


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

            if actual_samples > 0: # Ensure positive number of samples
                results_for_query[f"ContextCite{actual_samples}"] = harness.compute_contextcite_weights(num_samples=actual_samples, lasso_alpha=0.0, seed=SEED)
                
                results_for_query[f"WSS{actual_samples}"] = harness.compute_wss(num_samples=actual_samples, lasso_alpha=0.0, seed=SEED)

        for size_key, num_t in T_iterations_map.items():
            results_for_query[f"TMC{num_t}"] = harness.compute_tmc_shap(num_iterations=num_t, performance_tolerance=0.001, seed=SEED)
        
            if beta_dist: # beta_dist needs to be defined or imported (e.g., from scipy.stats)
                results_for_query[f"BetaShap (U){num_t}"] = harness.compute_beta_shap(num_iterations=num_t, beta_a=0.5, beta_b=0.5, seed=SEED)

        results_for_query["LOO"] = harness.compute_loo()

        exact_scores = results_for_query.get("Exact")
        # print("TYPE EXACT SHAP OUTPUT: ", type(exact_scores))
        # df_exact_shap.loc[i] = [query, list(exact_scores), list(flags)]
        df_exact_shap.loc[i, "query"] = query
        df_exact_shap.loc[i, "scores"] = np.array(exact_scores)
        df_exact_shap.loc[i, "doc_flags"] = flags
        print("RESULT EXACT SHAPE ", df_exact_shap)
        # print("EXACT SCORES: ", exact_scores)
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
                        else:
                            pearson_c, _ = pearsonr(exact_scores, approx_scores) #pearson measures linear correlation
                            spearman_c, _ = spearmanr(exact_scores, approx_scores) # spearman measures rank correlation (how well the order matches)

                            exact_ranks = rankdata(-np.array(exact_scores), method="average") # rank scores with the smallest =1 and when there is a tie assign the average rank
                            approx_ranks = rankdata(-np.array(approx_scores), method = "average")
                            kendall_c, _ = kendalltau(exact_ranks, approx_ranks) # return tau and pval (if pval is < 0.005 we can say that correlation is statistically significant) 
                        
                        ndgc_scoring  = ndcg_score(
                            [positive_exact_score], 
                            [approx_scores],
                            k = 3 # focus on top k document scoring
                        )
                        
                        all_metrics_data.append({
                            "Question_Index": i, "Query": query, "Method": method,
                            "Pearson": pearson_c, "Spearman": spearman_c, "NDCG" : ndgc_scoring, "KendallTau" : kendall_c,
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
df_exact_shap.to_csv("scripts/Exact_shap_details.csv", index=False)

# Save
df_exact_shap.to_json("scripts/Exact_shap_detail.json", orient="records", lines=True)

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
            Num_Valid_Queries=("Query", "nunique")
        ).sort_values(by="Avg_Spearman", ascending=False)
        
        print(average_metrics.round(4))

        details_path = f"Experiment_data/{DATASET_NAME}/{MODEL_NAME}/shapley_rag_experiment_details3bcp.csv"
        summary_path = f"Experiment_data/{DATASET_NAME}/{MODEL_NAME}/shapley_rag_experiment_summary3bcp.csv"
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