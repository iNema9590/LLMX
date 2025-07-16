import sys
import os
import random
import gc
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr, kendalltau, rankdata
from sklearn.metrics import ndcg_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

# Setup
start = time.time()
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from SHapRAG import ShapleyExperimentHarness

# docs = [
#     "Vitamin B1, also known as thiamine, is essential for glucose metabolism and neural function.",  # 🔑 Useful
#     "Chronic alcoholism can impair nutrient absorption, particularly leading to thiamine deficiency.",  # 🔑 Useful
#     "Vitamin C deficiency leads to scurvy, which presents with bleeding gums and joint pain.",
#     "Vitamin D deficiency is associated with rickets in children and osteomalacia in adults.",
#     "Vitamin B12 deficiency can cause neurological symptoms but is more common in strict vegans.",
#     "Folic acid is important for DNA synthesis and is crucial during pregnancy.",
#     "Vitamin A deficiency primarily affects vision and immune function.",
#     "Iron deficiency is the leading cause of anemia worldwide.",
#     "Calcium is essential for bone health and muscle contraction.",
#     "Vitamin K is important for blood clotting."
# ]
# query = "Which vitamin deficiency can lead to neurological symptoms and is commonly seen in chronic alcoholics?"
docs = [

"Paris is the capital of France.",
"The weather in Paris is sunny today.",
# "The sun is shining in Paris today",
"Berlin is the capital of Germany.", # Irrelevant
# "The Eiffel Tower is located in Paris, France.",
# "France borders several countries including Germany.",
"The currency used in France is the Euro.",
"Paris hosted the Summer Olympics in 1900 and 1924.",
"Germany uses the Euro as well.", # Redundant info
# "It is cloudy in Berlin today."
]
query = "What is the weather like in the capital of France?"
# Parameters
NUM_RETRIEVED_DOCS = len(docs)
SEED = 42

# Initialize Accelerator
accelerator_main = Accelerator(mixed_precision="fp16")

# Load Model
if accelerator_main.is_main_process:
    print("Main Script: Loading model...")
model_path = "meta-llama/Llama-3.2-3B-Instruct"
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

# Initialize Harness
harness = ShapleyExperimentHarness(
    items=docs,
    query=query,
    prepared_model_for_harness=prepared_model,
    tokenizer_for_harness=tokenizer,
    accelerator_for_harness=accelerator_main,
    verbose=True,
    utility_path=None
)

results_for_query = {}
all_metrics_data = []

# Compute metrics
if accelerator_main.is_main_process:
    results_for_query["Exact"] = harness.compute_exact_shap()

    m_samples_map = {"S": 32, "M": 64, "L": 100}
    T_iterations_map = {"S": 10, "M": 15, "L": 20}

    for size_key, num_s in m_samples_map.items():
        actual_samples = max(1, min(num_s, 2 ** len(docs) - 1))

        if actual_samples > 0:
            results_for_query[f"ContextCite{actual_samples}"] = harness.compute_contextcite_weights(
                num_samples=actual_samples, sampling="kernelshap", seed=SEED)
            results_for_query[f"WSS{actual_samples}"] = harness.compute_wss(num_samples=actual_samples, seed=SEED, distil=None, sampling="kernelshap",sur_type="gam", util='pure-surrogate', pairchecking=False)

            results_for_query[f"BetaShap (U){actual_samples}"] = harness.compute_beta_shap(
                num_iterations_max=T_iterations_map[size_key], beta_a=0.5, beta_b=0.5,
                max_unique_lookups=actual_samples, seed=SEED)
            results_for_query[f"TMC{actual_samples}"] = harness.compute_tmc_shap(
                num_iterations_max=T_iterations_map[size_key], performance_tolerance=0.001,
                max_unique_lookups=actual_samples, seed=SEED)

    results_for_query["LOO"] = harness.compute_loo()

    # Evaluate metrics
    exact_scores = results_for_query.get("Exact")
    if exact_scores is not None:
        positive_exact_score = np.clip(exact_scores, a_min=0.0, a_max=None)
        for method, approx_scores in results_for_query.items():
            if method != "Exact" and approx_scores is not None and len(approx_scores) == len(exact_scores):
                if np.all(exact_scores == exact_scores[0]) or np.all(approx_scores == approx_scores[0]):
                    pearson_c = spearman_c = 1.0 if np.allclose(exact_scores, approx_scores) else 0.0
                else:
                    pearson_c, _ = pearsonr(exact_scores, approx_scores)
                    spearman_c, _ = spearmanr(exact_scores, approx_scores)
                    exact_ranks = rankdata(-np.array(exact_scores), method="average")
                    approx_ranks = rankdata(-np.array(approx_scores), method="average")
                    kendall_c, _ = kendalltau(exact_ranks, approx_ranks)
                ndgc_scoring = ndcg_score([positive_exact_score], [approx_scores], k=3)

                all_metrics_data.append({
                     "Method": method,
                    "Pearson": pearson_c, "Spearman": spearman_c, "NDCG": ndgc_scoring, "KendallTau": kendall_c
                })
if all_metrics_data:
    metrics_df_all_questions = pd.DataFrame(all_metrics_data)
    
    print("\n\n============================")
    print("     Correlation Metrics")
    print("============================")
    print(metrics_df_all_questions.round(4).to_string(index=False))
    print("\n\n============================")
    print("     Approximate Scores")
    print("============================")
    for method, approx_scores in results_for_query.items():
        if approx_scores is not None:
            print(f"\nMethod: {method}")
            print(np.round(approx_scores, 4))
accelerator_main.wait_for_everyone()
del harness

if torch.cuda.is_available():
    if accelerator_main.is_main_process:
        print(f"Attempting to empty CUDA cache on rank {accelerator_main.process_index} ")
    torch.cuda.empty_cache()
    gc.collect()
    if accelerator_main.is_main_process:
        print(f"CUDA cache empty attempt complete on rank {accelerator_main.process_index}.")
accelerator_main.wait_for_everyone()


# Final cleanup
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
        print(f"Rank {accelerator_main.process_index} (Local Main): Distributed environment not initialized or not available.")

if accelerator_main.is_main_process:
    print("Script fully exited.")
end = time.time()
print(f"Execution time: {end - start:.4f} seconds")