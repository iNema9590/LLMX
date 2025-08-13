import sys
import os
import random
import gc
import time
import torch
import numpy as np
import pandas as pd
import ast
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr, kendalltau, rankdata
from sklearn.metrics import ndcg_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
# os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from SHapRAG import *
starttime=time.time()
df= pd.read_csv("../data/synthetic_data/20_synergy_hard_negatives.csv",index_col=False, sep=";")

SEED = 42
# Initialize Accelerator
accelerator_main = Accelerator(mixed_precision="fp16")

# Load Model
if accelerator_main.is_main_process:
    print("Main Script: Loading model...")
# model_path = "mistralai/Mistral-7B-Instruct-v0.3"
model_path = "meta-llama/Llama-3.1-8B-Instruct"
# model_path = "Qwen/Qwen2.5-3B-Instruct"

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
# num_questions_to_run=50
k_values = [1,2,3,4,5]
all_results=[]
LDSs=[]
r2_fm=[]
r2_cc=[]
for i in tqdm(range(num_questions_to_run), disable=not accelerator_main.is_main_process):
    query = df.question[i]
    if accelerator_main.is_main_process:
        print(f"\n--- Question {i+1}/{num_questions_to_run}: {query[:60]}... ---")

    p=ast.literal_eval(df.context[i])
    docs=p[:5]+ [p[1]] + p[5:]
    utility_cache_base_dir = f"../Experiment_data/synthetic/{model_path.split('/')[1]}/duplicates"
    utility_cache_filename = f"utilities_q_idx{i}.pkl" # More robust naming
    current_utility_path = os.path.join(utility_cache_base_dir, utility_cache_filename)
    
    if accelerator_main.is_main_process: # Only main process creates directories
        os.makedirs(os.path.dirname(current_utility_path), exist_ok=True)
    
    # Initialize Harness
    harness = ContextAttribution(
        items=docs,
        query=query,
        prepared_model=prepared_model,
        prepared_tokenizer=tokenizer,
        accelerator=accelerator_main,
        utility_cache_path=current_utility_path
    )

    print(f'Response: {harness.target_response}')
    print(f'GT: {df.answer[i]}')
    # Compute metrics
    results_for_query = {}
    if accelerator_main.is_main_process:
        m_samples_map = {"L": 364} 
        # m_samples_map = {"L": 128, "XL":256, "XXL":512} 
        T_iterations_map = {"L":40, "XL":50, "XXL":60} 

        for size_key, num_s in m_samples_map.items():
            if 2**len(docs) < num_s and size_key != "L":
                actual_samples = max(1, 2**len(docs)-1 if 2**len(docs)>0 else 1)
            else:
                actual_samples = num_s

            if actual_samples > 0:
                results_for_query[f"ContextCite{actual_samples}"], model_cc = harness.compute_contextcite(num_samples=actual_samples, seed=SEED)
                attributions, ints=harness.compute_spex(sample_budget=actual_samples,max_order=2)
                results_for_query[f"FBII{actual_samples}"]=attributions['fbii']
                results_for_query[f"Spex{actual_samples}"]=attributions['fourier']
                results_for_query[f"FSII{actual_samples}"]=attributions['fsii']
                results_for_query[f"FM_WeightsD{actual_samples}"], F, modelfm = harness.compute_wss(num_samples=actual_samples, seed=SEED, sampling="kernelshap",sur_type="fm", utility_mode="divergence_utility")
                results_for_query[f"FM_Weights{actual_samples}"], F, _ = harness.compute_wss(num_samples=actual_samples, seed=SEED, sampling="kernelshap",sur_type="fm")
                # results_for_query[f"BetaShap{actual_samples}"] = harness.compute_beta_shap(num_iterations_max=T_iterations_map[size_key], beta_a=16, beta_b=1, max_unique_lookups=actual_samples, seed=SEED)
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
                                            utility_type="divergence"
                                        )
        
        r2_fm.append([harness.r2_mse(30, 'divergence_utility', modelfm, method='fm')])
        r2_cc.append([harness.r2_mse(30, 'logit-prob', model_cc, method='cc')])

        LDS = {}
        for j in results_for_query:
            calculate_LDS = {j:harness.lds(results_for_query[j], 30)}
            LDS.update(calculate_LDS)
        # LDS = [{j:harness.lds(results_for_query[j], 30)} for j in results_for_query]

        results_for_query["topk_probability"] = prob_topk
        results_for_query["topk_divergence"] = div_topk
        results_for_query["LDS"] = LDS
        harness.save_utility_cache(current_utility_path)
        
        all_results.append(results_for_query)
endtime=time.time()
print(f'It took - {endtime-starttime}')
with open(utility_cache_base_dir, 'wb') as f:
    pickle.dump(all_results, f)