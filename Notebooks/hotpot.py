import sys
import os
import random
import gc
import time
import torch
import numpy as np
import pandas as pd
import pickle
import ast
from tqdm import tqdm
from scipy.sparse import csr_matrix
import itertools
from scipy.stats import spearmanr, pearsonr, kendalltau, rankdata
from sklearn.metrics import ndcg_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
# import nltk
# nltk.download('punkt')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from SHapRAG import *
from SHapRAG.utils import *

dfin=pd.read_csv("../data/sampled_hotpot.csv")
dfin.reordered_sentences=dfin.reordered_sentences.apply(ast.literal_eval)
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
utility_cache_base_dir = f"../Experiment_data/sampled_hotpot/{model_path.split('/')[1]}"
accelerator_main.wait_for_everyone()

num_questions_to_run = len(dfin)
K_VALUES = [1, 2, 3, 4, 5]
all_results = []
extras = []
def GT(i):
    if dfin["len_gt"][i]==2:
        return [0,1]
    elif dfin["len_gt"][i]==3:
        return [0,1,2]
    elif dfin["len_gt"][i]==4:
        return [0,1,2,3]

resposes = []
for i in range(num_questions_to_run):
    query = dfin.question[i]
    if accelerator_main.is_main_process:
        print(f"\n--- Question {i+1}/{num_questions_to_run}: {query[:60]}... ---")

    docs = dfin.reordered_sentences[i][:10]
    utility_cache_filename = f"utilities_q_idx{i}.pkl"
    current_utility_path = os.path.join(utility_cache_base_dir, utility_cache_filename)

    if accelerator_main.is_main_process:
        os.makedirs(os.path.dirname(current_utility_path), exist_ok=True)

    harness = ContextAttribution(
        items=docs,
        query=query,
        prepared_model=prepared_model,
        prepared_tokenizer=tokenizer,
        accelerator=accelerator_main,
        utility_cache_path=current_utility_path,
        utility_mode='log-perplexity'
    )
    full_budget=pow(2,harness.n_items)
    print(f'Target response: {harness.target_response} - GT: {dfin.answer[i]})')
    # res = evaluate(dfin.question[i], harness.target_response, dfin.answer[i])
    # resposes.append(res)
    if accelerator_main.is_main_process:
        methods_results = {}
        metrics_results = {}
        extra_results = {}

        m_samples_map = {"XS":32, "S":64, "M":128, "L":264, "XL":528, "XXL":724}
        fm_models = {}
        methods_results['Exact-Shapley']=harness._calculate_exact(method='SV')
        for size_key, actual_samples in m_samples_map.items():
            print(f"Running sample size: {actual_samples}")
            methods_results[f"ContextCite_{actual_samples}"], fm_models[f"ContextCite_{actual_samples}"] = harness.compute_contextcite(
                num_samples=actual_samples, seed=SEED
            )

            methods_results[f"FACILE_{actual_samples}"], extra_results[f"FACILE_{actual_samples}"], fm_models[f"FACILE_{actual_samples}"] = harness.compute_wss(
                    num_samples=actual_samples,
                    seed=SEED,
                    sampling_method="bf_kernelshap",
                    sur_type="fm_tuning",
                    selection_metric="r2_delta")
            
            attributionshapiq, interactionshapiq, fm_models[f"Shapiq_{actual_samples}"] = harness.compute_shapiq(budget=actual_samples, method='FSII')
            methods_results[f"Shapiq_{actual_samples}"] = attributionshapiq
            extra_results.update({
                f"Shapiq_{actual_samples}":interactionshapiq
                                                                        })
            try:
                attributionshap, interactionshap, fm_models[f"Spex_{actual_samples}"] = harness.compute_spex(sample_budget=actual_samples, max_order=harness.n_items, method='FSII')
                methods_results[f"Spex_{actual_samples}"] = attributionshap

                extra_results.update({
                f"Spex_{actual_samples}":interactionshap
                                                                        })
            except Exception: 
                pass

            try:
                attributionban, interactionban, fm_models[f"ProxySpex_{actual_samples}"] = harness.compute_proxyspex(sample_budget=actual_samples, max_order=harness.n_items, method='FSII')
                methods_results[f"ProxySpex_{actual_samples}"] = attributionban
                extra_results.update({
                    f"ProxySpex_{actual_samples}":interactionban
                                                                        })

            except Exception: 
                pass

        attributionxs, interactionxs, fm_models["Exact-FSII"] = harness.compute_exact_faith(max_order=2, method='FSII')

        extra_results.update({
        "Exact-FSII": interactionxs
    })
        methods_results["Exact-FSII"]=attributionxs

        # --- Evaluation Metrics ---
        metrics_results["topk_probability"] = harness.evaluate_topk_performance(
            methods_results, fm_models, K_VALUES
        )

        # R² and Delta R²
        metrics_results["R2"] = harness.r2(methods_results,50, models=fm_models)
        metrics_results["Delta_R2"] = harness.delta_r2(methods_results,20, models=fm_models)
        metrics_results['Recall']=harness.recall_at_k(GT(i), methods_results, K_VALUES)

        # LDS per method
        metrics_results["LDS"] = harness.lds(methods_results,50, models=fm_models)
        all_results.append({
            "query_index": i,
            "query": query,
            "ground_truth": dfin.answer[i],
            "response": harness.target_response,
            "methods": methods_results,
            "metrics": metrics_results
        })
        extras.append(extra_results)

        # Save utility cache
        harness.save_utility_cache(current_utility_path)

with open(f"{utility_cache_base_dir}/results.pkl", "wb") as f:
    pickle.dump(all_results, f)

with open(f"{utility_cache_base_dir}/extras.pkl", "wb") as f:
    pickle.dump(extras, f)


# with open(f"{utility_cache_base_dir}/responses.pkl", "wb") as f:
#     pickle.dump(resposes, f)            
