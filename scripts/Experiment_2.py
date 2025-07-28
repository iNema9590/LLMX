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
#from utils import hit_rate_at_k, precision_at_k, reciprocal_rank

from SHapRAG import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, required=True)
parser.add_argument("--precision", default='bf16', type = str)
parser.add_argument("--model_name", type = str, required = True)
parser.add_argument("--shuffle", action="store_true")
args = parser.parse_args()

available_models = {"qwen_3B" : "Qwen/Qwen2.5-3B-Instruct", 
                    "llama_3B" : "meta-llama/Llama-3.2-3B-Instruct", 
                    "llama_8B" : "meta-llama/Llama-3.1-8B-Instruct", 
                    "mistral_7B" : "mistralai/Mistral-7B-Instruct-v0.3"}

start = time.time()

# Construct full path to CSV file in same folder

#csv_path = os.path.join(current_dir, f'../data/{args.dataset}.csv')
csv_path = os.path.join(f'../data/{args.dataset}.csv')

df = pd.read_csv(csv_path)
df_save_results = pd.DataFrame(columns = ["query", "context", "provided_answer",
                                           "lds", "rmse", "r2",
                                           "topk_probability", "topk_divergence"])
print("Data Loaded!")

num_questions_to_run = 100
print(f"Running experiments for {num_questions_to_run} questions...")

# Parameters
SEED = 42
DATASET_NAME = f"{args.dataset}"
PRECISION = f"{args.precision}"
SHUFFLE = ["SHUFFLE" if args.shuffle else "VANILLA"][0]
random.seed(42) # SEED SO THAT WE CAN REPLICATE IF NEEDED
top_k = [1, 3, 5]

if args.shuffle: 
    index_list = list(range(10)) # context size

# Initialize Accelerator ONCE
accelerator_main = Accelerator(mixed_precision=PRECISION)

if accelerator_main.is_main_process:
    print(f"Main Script: Loading model...")
model_path = available_models[args.model_name]
# MODEL_NAME = model_path.split('/')[1]
MODEL_NAME = args.modle_name

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

all_queries_data = []
k_values = [1, 2]
LDSs=[]
r2s = []
RMSEs = []

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

    if args.shuffle == True: 
        index_list = list(range(10))
        random.seed(i) # different seed for each query/line
        random.shuffle(index_list)
        docs = [docs[j] for j in index_list]
        flags = [flags[j] for j in index_list]

    utility_cache_base_dir = f"../Experiment_data/{DATASET_NAME}/{MODEL_NAME}/{SHUFFLE}/utilities_cache3bcp"
    utility_cache_filename = f"utilities_q_idx{i}_n{len(docs)}.pkl" # More robust naming
    current_utility_path = os.path.join(utility_cache_base_dir, utility_cache_filename)
    
    if accelerator_main.is_main_process: # Only main process creates directories
        os.makedirs(os.path.dirname(current_utility_path), exist_ok=True)
        print(f"  Instantiating ShapleyExperimentHarness for Q{i} (n={len(docs)} docs)...")
    
    # Synchronize before creating harness instance to ensure directory exists for all if utility_path is used by others
    # accelerator_main.wait_for_everyone() 
    harness = ContextAttribution(
        items=docs,
        query=query,
        prepared_model_for_harness=prepared_model,
        tokenizer_for_harness=tokenizer,
        accelerator_for_harness=accelerator_main,
        utility_cache_path=current_utility_path
    )

    print(f'Response: {harness.target_response}')
            
    results_for_query = {}
    if accelerator_main.is_main_process:
        m_samples_map = {"S": 128, "M": 256, "L": 512} # Example sample sizes
        T_iterations_map = {"L":40}  # Example iterations

    for size_key, num_s in m_samples_map.items():
        if 2**len(docs) < num_s and size_key != "L": # Avoid sampling more than possible, except for a default
            actual_samples = max(1, 2**len(docs)-1 if 2**len(docs)>0 else 1) # Ensure at least 1 sample if possible
        else:
            actual_samples = num_s

        if actual_samples > 0: 
            results_for_query[f"ContextCite{actual_samples}"], model_cc = harness.compute_contextcite(num_samples=actual_samples, seed=SEED) #lasso_alpha=0.01,
        # results_for_query[f"FM_Shap{actual_samples}"], _, F, modelfm = harness.compute_wss(num_samples=actual_samples, seed=SEED, sampling="kernelshap",sur_type="fm", k=4)

        # results_for_query[f"KernelShap{actual_samples}"] = harness.compute_contextcite_weights(num_samples=actual_samples, sampling="kernel_shap",  seed=SEED) #lasso_alpha=0.01,
        # results_for_query[f"WSS{actual_samples}"] = harness.compute_wss(num_samples=actual_samples,  seed=SEED, sampling="uniform") #lasso_alpha=0.01,
        # results_for_query[f"BetaShap (U){actual_samples}"] = harness.compute_beta_shap(num_iterations_max=T_iterations_map[size_key], beta_a=0.5, beta_b=0.5, max_unique_lookups=actual_samples, seed=SEED)
        # results_for_query[f"TMC{actual_samples}"] = harness.compute_tmc_shap(num_iterations_max=T_iterations_map[size_key], performance_tolerance=0.001, max_unique_lookups=actual_samples, seed=SEED)
    
    results_for_query["LOO"] = harness.compute_loo()
    results_for_query["ARC-JSD"] = harness.compute_arc_jsd()


    prob_topk = harness.evaluate_topk_performance(
                                            results_for_query, 
                                            k_values, 
                                            utility_type="probability"
                                        )

    div_topk = harness.evaluate_topk_performance(
                                        results_for_query, 
                                        k_values, 
                                        utility_type="divergence",
    )
                                    


    # exact_scores = results_for_query.get("Exact")

    LDS = []
    r2 = []
    RMSE = []
    for i in results_for_query:
        if "FM_Shap" in i:
            calculate_LDS = {i:harness.lds(results_for_query[i], 30, utl=True, model=modelfm)}
            calculate_r2 = {i:harness.r2(30, model=modelfm)}
            calculate_RMSE = {i:harness.RMSE(30, model=modelfm)}
            LDS.append(calculate_LDS)
            r2.append(calculate_r2)
            RMSE.append(calculate_RMSE)
        elif "ContextCite" in i:
            calculate_LDS = {i:harness.lds(results_for_query[i], 30)}
            calculate_r2 = {i:harness.r2(30, model=model_cc, method="notfm")}
            calculate_RMSE = {i:harness.RMSE(30, model=model_cc, method="notfm")}
            LDS.append(calculate_LDS)
            r2.append(calculate_r2)
            RMSE.append(calculate_RMSE)
    # #LDS = [{i:harness.lds(results_for_query[i], 30)} for i in results_for_query]
    # LDSs.append(LDS)
    # r2s.append(r2)
    # RMSEs.append(RMSE)
    df_save_results.loc[i, "query"] = query
    df_save_results.loc[i, "lds"] = [LDS]
    df_save_results.loc[i, "r2"] = [r2]
    df_save_results.loc[i, "rmse"] = [RMSE]
    df_save_results.loc[i, "topk_probability"] = prob_topk
    df_save_results.loc[i, "topk_divergence"] = div_topk
        # df_save_results.loc[i, "top_k"] = [top_k_docs]
        # df_save_results.loc[i, "methods_top_k"] = [methods_top_k]
        # df_save_results.loc[i, "precision_top_k"] = [precision_top_k]
        # print("Flags Value: ", flags)
        # df_save_results.loc[i, "doc_id"] = [[flags]]
    df_save_results.loc[i, "context"] = [[docs]]
    df_save_results.loc[i, "provided_answer"] = harness.target_response


        #print("RESULTS FOR QUERY: ", df_save_results)
        # all_queries_data.append(precision_top_k)
    
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

all_metrics = {}
result_df = {}
for row in df_save_results['precision_top_k']:
   for method in row[0]:
       try:
           all_metrics[method].append(row[0][method])
       except:
           all_metrics[method] = [row[0][method]]
for method in all_metrics:
    result_df[method] = round(np.mean(all_metrics[method]), 4)
    
processed_data = {}
for method, value in result_df.items():
    method_name = method[1]
    col_num = method[0]
    if method_name not in processed_data:
        processed_data[method_name] = {}
    processed_data[method_name][col_num] = value

# Create a Pandas DataFrame from the processed data
resulting_df = pd.DataFrame(processed_data).T
print(resulting_df)


resulting_df.to_csv(f"../Experiment_data/{DATASET_NAME}/{MODEL_NAME}/results_top_k.csv", index=False)

# Save
resulting_df.to_json(f"../Experiment_data/{DATASET_NAME}/{MODEL_NAME}/results_top_k.json", orient="records", lines=True)

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