# main_experiment.py
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Import your custom class
import sys
import os
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from SHapRAG import*
# It's good practice to have an if __name__ == "__main__": block
# even when using accelerate launch, as it clarifies the entry point for each process.
if __name__ == "__main__":
    # Model and tokenizer loading (this happens in each process)
    # It's crucial that the model is loaded in a way that `accelerator.prepare()` can handle it.
    # Loading on CPU first is generally safe.
    model_path = "meta-llama/Llama-3.2-3B-Instruct"  # Or your desired model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        # device_map="auto" could also be used if you prefer Transformers to handle initial placement,
        # but accelerator.prepare() will still manage the final device assignment per process.
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    items_for_shapley = [
    "The weather in Paris is sunny today.",
    "Paris is the capital of France.",
    # "The sun is shining in Paris today",
    "Berlin is the capital of Germany.", # Irrelevant
    # "The Eiffel Tower is located in Paris, France.",
    # "France borders several countries including Germany.",
    "The currency used in France is the Euro.",
    "Paris hosted the Summer Olympics in 1900 and 1924.",
    "Germany uses the Euro as well.", # Redundant info
    "It is cloudy in Berlin today."
]
    query_for_shapley = "What is the weather like in the capital of France?"

    # --- Harness Initialization and Usage ---
    # Each process will create an instance of ShapleyExperimentHarness.
    # The Accelerator instance within the harness will be process-specific.
    harness_start_time = time.time()
    harness = ShapleyExperimentHarness(
        items=items_for_shapley,
        query=query_for_shapley,
        model=model,         # Pass the loaded model
        tokenizer=tokenizer, # Pass the loaded tokenizer
        verbose=True
    )
    harness_init_done_time = time.time()

    # The `accelerator` instance is part of the harness
    current_accelerator = harness.accelerator

    # Only the main process typically prints results or performs final aggregations
    if current_accelerator.is_main_process:
        print(f"Harness initialized. Pre-computation took: {harness_init_done_time - harness_start_time:.2f} seconds.")
        print(f"Total pre-computed utilities: {len(harness.all_true_utilities)}")

        # Compute and print results
        exact_shapley_values = harness.compute_exact_shap()
        print("\nExact Shapley Values:", exact_shapley_values)

        contextcite_weights = harness.compute_contextcite_weights(num_samples=min(10, 2**harness.n_items), lasso_alpha=0.01)
        print("\nContextCite Weights:", contextcite_weights)
        
        # ... (call other methods as needed) ...
        
        loo_scores = harness.compute_loo()
        print("\nLOO Scores:", loo_scores)

        methods_done_time = time.time()
        print(f"\nTotal execution time for all methods after init: {methods_done_time - harness_init_done_time:.2f} seconds.")
        print(f"Overall script time: {methods_done_time - harness_start_time:.2f} seconds.")

    # Important: ensure all processes synchronize before exiting, especially if there are
    # background operations or if some processes might finish much earlier.
    current_accelerator.wait_for_everyone()

    # if current_accelerator.is_main_process:
    #     print("All processes finished.")