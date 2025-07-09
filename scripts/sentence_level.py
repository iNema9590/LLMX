import argparse
import ast
import json
import os
import sys

import nltk
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from nltk.tokenize import sent_tokenize
from scipy.stats import kendalltau, pearsonr, rankdata, spearmanr
from sklearn.metrics import ndcg_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from SHapRAG import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, required=True)
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument('--precision', default='bf16', type = 'str')
args = parser.parse_args()

available_models = {"qwen_3B" : "Qwen/Qwen2.5-3B-Instruct",
                    "llama_3B" : "meta-llama/Llama-3.2-3B-Instruct",
                    "llama_8B" : "meta-llama/Llama-3.1-8B-Instruct",
                    "mistral_7B" : "mistralai/Mistral-7B-Instruct-v0.3"}

csv_path = os.path.join(f'../data/{args.dataset}.csv')


df = pd.read_csv(csv_path)
print("Data Loaded!")



SEED = 42
# Initialize Accelerator
accelerator_main = Accelerator(mixed_precision="fp16")

# Load Model
if accelerator_main.is_main_process:
    print("Main Script: Loading model...")
# model_path = "mistralai/Mistral-7B-Instruct-v0.3"
# model_path = "meta-llama/Llama-3.1-8B-Instruct"
model_path = "Qwen/Qwen2.5-3B-Instruct"

model_cpu = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model_cpu.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model_cpu, 'generation_config') and model_cpu.generation_config is not None:
        model_cpu.generation_config.pad_token_id = tokenizer.pad_token_id

if accelerator_main.is_main_process:
    print("Main Script: Preparing model with Accelerator...")
prepared_model = accelerator_main.prepare(model_cpu)
unwrapped_prepared_model = accelerator_main.unwrap_model(prepared_model)
unwrapped_prepared_model.eval()
if accelerator_main.is_main_process:
    print("Main Script: Model prepared and set to eval.")

# Define utility cache
accelerator_main.wait_for_everyone()




num_questions_to_run=len(df.question)
# num_questions_to_run=1
all_metrics_data = []
all_results=[]
M=[]
Fs=[]
pairs=[]
mse_inters=[]
mse_lins=[]
mse_fms=[]

for i in tqdm(range(num_questions_to_run), desc="Processing Questions", disable=not accelerator_main.is_main_process):
    query = df.question[i]
    if accelerator_main.is_main_process:
        print(f"\n--- Question {i+1}/{num_questions_to_run}: {query[:60]}... ---")

    if isinstance(df.context[i], list) == False: 
        docs=ast.literal_eval(df.context[i])
    else: 
        docs = df.context[i]

    utility_cache_base_dir = f"../Experiment_data/{args.dataset_name}"
    utility_cache_filename = f"utilities_q_idx{i}_n{len(docs)}.pkl" # More robust naming
    current_utility_path = os.path.join(utility_cache_base_dir, utility_cache_filename)
    
    if accelerator_main.is_main_process: # Only main process creates directories
        os.makedirs(os.path.dirname(current_utility_path), exist_ok=True)
        print(f"  Instantiating ShapleyExperimentHarness for Q{i} (n={len(docs)} docs)...")
    
    # Initialize Harness
    harness = ContextAttribution(
        items=docs,
        query=query,
        prepared_model_for_harness=prepared_model,
        tokenizer_for_harness=tokenizer,
        accelerator_for_harness=accelerator_main,
        verbose=False
    )
    # Compute metrics
    results_for_query = {}
    # M.append(harness.compute_shapley_interaction_index_pairs_matrix())
    if accelerator_main.is_main_process:

        m_samples_map = {"L": 32}
        T_iterations_map = { "L":20}

        for size_key, num_s in m_samples_map.items():
            if 2**len(docs) < num_s and size_key != "L":
                actual_samples = max(1, 2**len(docs)-1 if 2**len(docs)>0 else 1)
            else:
                actual_samples = num_s
            
            if actual_samples > 0: 
                results_for_query[f"ContextCite{actual_samples}"] = harness.compute_contextcite(num_samples=actual_samples, seed=SEED)
                print("HERE")

                results_for_query[f"WSS_FM{actual_samples}"], F, mse_fm = harness.compute_wss(num_samples=actual_samples, seed=SEED)
                Fs.append(F)
                mse_fms.append(mse_fm)
                results_for_query[f"BetaShap{actual_samples}"] = harness.compute_beta_shap(num_iterations_max=T_iterations_map[size_key], beta_a=4, beta_b=4, max_unique_lookups=actual_samples, seed=SEED)
                results_for_query[f"TMC{actual_samples}"] = harness.compute_tmc_shap(num_iterations_max=T_iterations_map[size_key], performance_tolerance=0.001, max_unique_lookups=actual_samples, seed=SEED)

        results_for_query["LOO"] = harness.compute_loo()
        results_for_query["ARC-JSD"] = harness.compute_arc_jsd()

        # exact_scores = results_for_query.get("ExactInter")
        all_results.append(results_for_query)
    

np.save(f'../Experiment_data/{args.dataset_name}/{args.model_name}/all_results.npy', all_results)





