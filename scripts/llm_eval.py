import sys
import os

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

import math
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
import numpy as np
from accelerate import Accelerator
import pandas as pd
from tqdm import tqdm
import gc
import time
import ast

# --- Configuration ---
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
#MODEL_NAME = "gpt2"
# For faster testing with a smaller model, you could use "gpt2" or "distilgpt2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dataset(dataset_name='bioasq'):
    if dataset_name == 'bioasq':
        splits = {'train': 'question-answer-passages/train-00000-of-00001.parquet', 'test': 'question-answer-passages/test-00000-of-00001.parquet'}
        df = pd.read_parquet("hf://datasets/enelpol/rag-mini-bioasq/" + splits["train"])

        df1 = pd.read_parquet("hf://datasets/enelpol/rag-mini-bioasq/text-corpus/test-00000-of-00001.parquet")
        df1.set_index('id', inplace=True)

        df1['passage']=df1['passage'].str.replace(r'[\n]', ' ', regex=True)
        df['question']=df['question'].str.replace(r'[\n]', ' ', regex=True)
        df['answer']=df['answer'].str.replace(r'[\n]', ' ', regex=True)
        df = df[df['relevant_passage_ids'].apply(len) >= 10].reset_index(drop=True)
        return df, df1

def llm_generate_response(tokenizer, device, model, query, accelerator, context_str: str, max_new_tokens: int = 200) -> str:
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    if context_str:
        messages.append({"role": "user", "content": f"Given the context: {context_str}. Briefly answer the query: {query}"})
    else:
        messages.append({"role": "user", "content": query})

    chat_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
        )
    tokenized = tokenizer(chat_text, return_tensors="pt", padding=True)

    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    unwrapped_model = accelerator.unwrap_model(model)
    generated_ids = None # Initialize
    outputs_dict = None # Initialize for potential outputs if model returns dict

    with torch.no_grad():
     
        outputs_gen = unwrapped_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None \
            else unwrapped_model.config.eos_token_id,
            temperature=1.0,
            top_p=1.0,
            )
            
        # Handle different return types of generate
        if isinstance(outputs_gen, torch.Tensor):
            generated_ids = outputs_gen
        elif isinstance(outputs_gen, dict) and "sequences" in outputs_gen: # Common for GenerateOutput
            generated_ids = outputs_gen["sequences"]
        elif hasattr(outputs_gen, 'sequences'): # For GenerateOutput like objects
            generated_ids = outputs_gen.sequences
        else:
            # Fallback or error if unexpected output
            print(f"Warning: Unexpected output type from model.generate: {type(outputs_gen)}")
            # Try to find a tensor that looks like token IDs
            if isinstance(outputs_gen, (list, tuple)) and len(outputs_gen) > 0 and isinstance(outputs_gen[0], torch.Tensor):
                generated_ids = outputs_gen[0] # Guessing the first tensor is it
            else: # Cannot determine, return empty or raise error
                del input_ids, attention_mask, unwrapped_model, outputs_gen
                torch.cuda.empty_cache() # Attempt to clear if things are really messy
                return ""


    response_text = tokenizer.decode(generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    cleaned_text = response_text.lstrip().removeprefix("assistant").lstrip(": \n").strip()

    # Explicitly delete tensors
    del input_ids, attention_mask, generated_ids, unwrapped_model, outputs_gen
    torch.cuda.empty_cache()
    return cleaned_text


def setup_models_and_tokenizer(model_name: str):
    """
    Initializes the tokenizer and models.
    """
    
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set pad token if not set, common for Llama models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set tokenizer.pad_token to tokenizer.eos_token ({tokenizer.eos_token})")


    print(f"Loading model for embeddings ({model_name})...")
    try:
        # For embeddings, we typically use the base model (AutoModel)
        embedding_model = AutoModel.from_pretrained(model_name).to(DEVICE)
        embedding_model.eval() # Set to evaluation mode
    except Exception as e:
        print(f"Error loading embedding model {model_name}: {e}")
        print("Falling back to using CausalLM model for embeddings if it loads.")
        # If AutoModel fails, try to use AutoModelForCausalLM if it loads,
        # though AutoModel is generally preferred for just getting embeddings.
        embedding_model = None # Reset to ensure it's re-assigned or error is clear

    print(f"Loading model for Causal LM (log-likelihood) ({model_name})...")
    causal_lm_model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    causal_lm_model.eval() # Set to evaluation mode
    if embedding_model is None: # Fallback if AutoModel failed
        embedding_model = causal_lm_model 
        print("Using CausalLM model for embeddings as AutoModel failed to load.")

    
    if tokenizer is None or embedding_model is None or causal_lm_model is None:
        raise RuntimeError("Failed to load one or more essential models/tokenizer. Please check MODEL_NAME and access.")

    print("Models and tokenizer loaded successfully.")
    return tokenizer, embedding_model, causal_lm_model


def calculate_token_match_percentage(tokenizer, embedding_model, causal_lm_model, generated_text: str, target_text: str) -> float:
    """
    Calculates the percentage of tokens in the target text that are identically
    matched by tokens in the generated text at the same positions.
    Uses Hugging Face AutoTokenizer.

    Args:
        generated_text: The text generated by the LLM.
        target_text: The reference target text.

    Returns:
        A float between 0.0 and 100.0 representing the percentage of matching tokens.
    """
    if tokenizer is None:
        raise RuntimeError("Tokenizer not initialized. Call setup_models_and_tokenizer() first.")

    generated_tokens = tokenizer.tokenize(generated_text)
    target_tokens = tokenizer.tokenize(target_text)

    if not target_tokens:
        return 100.0 if not generated_tokens else 0.0 # Both empty or target empty

    matches = 0
    for i in range(min(len(generated_tokens), len(target_tokens))):
        if generated_tokens[i] in target_tokens:
            matches += 1
    
    percentage = (matches / len(target_tokens)) * 100.0
    return percentage

def get_sentence_embedding(text: str, model, hf_tokenizer) -> np.ndarray:
    """
    Computes sentence embedding using mean pooling of last hidden states.
    """
    if not text.strip(): # Handle empty string
        # Return a zero vector of the expected dimension if possible, or handle error
        # Assuming hidden size is accessible; might need a more robust way for unknown models
        try:
            hidden_size = model.config.hidden_size
            return np.zeros((hidden_size,))
        except AttributeError:
             print("Warning: Could not determine hidden_size for empty string embedding. Returning small zero vector.")
             return np.zeros((1,)) # Fallback, might cause dimension mismatch later

    inputs = hf_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # Use last hidden state. For some models, pooler_output might be available and preferred.
        # Mean pooling over token embeddings.
        last_hidden_states = outputs.hidden_states[-1]
        sentence_embedding = torch.mean(last_hidden_states, dim=1).squeeze().cpu().numpy()
    return sentence_embedding

def calculate_embedding_cosine_similarity(tokenizer, embedding_model, causal_lm_model, generated_text: str, target_text: str) -> float:
    """
    Calculates cosine similarity based on sentence embeddings from a Transformer model.
    """
    if embedding_model is None or tokenizer is None:
        raise RuntimeError("Embedding model or tokenizer not initialized. Call setup_models_and_tokenizer() first.")

    if not generated_text.strip() and not target_text.strip():
        return 1.0 # Both effectively empty
    if not generated_text.strip() or not target_text.strip():
        return 0.0 # One is empty, the other is not

    generated_embedding = get_sentence_embedding(generated_text, embedding_model, tokenizer)
    target_embedding = get_sentence_embedding(target_text, embedding_model, tokenizer)
    
    # Ensure embeddings are 2D for sklearn's cosine_similarity
    if generated_embedding.ndim == 1:
        generated_embedding = generated_embedding.reshape(1, -1)
    if target_embedding.ndim == 1:
        target_embedding = target_embedding.reshape(1, -1)

    if generated_embedding.shape[1] != target_embedding.shape[1]:
        print(f"Warning: Embedding dimensions mismatch. Gen: {generated_embedding.shape}, Target: {target_embedding.shape}. Returning 0.0 similarity.")
        # This can happen if one text results in an empty embedding (e.g., only PAD tokens after truncation)
        # or if the fallback zero vector for empty strings had a different dimension.
        return 0.0

    similarity = sklearn_cosine_similarity(generated_embedding, target_embedding)[0, 0]
    return float(similarity)


def calculate_log_likelihood(tokenizer, embedding_model, causal_lm_model, text_to_evaluate: str) -> float:
    """
    Calculates the total log-likelihood of the `text_to_evaluate` using a Causal LM.
    The score is sum(log P(token_i | token_0, ..., token_{i-1})).
    A higher (less negative) score is better.

    Args:
        text_to_evaluate: The text whose log-likelihood is to be calculated.

    Returns:
        The total log-likelihood of the text. Returns -inf if text is empty or
        if an error occurs.
    """
    if causal_lm_model is None or tokenizer is None:
        raise RuntimeError("Causal LM model or tokenizer not initialized. Call setup_models_and_tokenizer() first.")

    if not text_to_evaluate.strip():
        return 0.0 # Or -float('inf') depending on convention for empty sequence likelihood

    inputs = tokenizer(text_to_evaluate, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    input_ids = inputs.input_ids
    
    if input_ids.shape[1] == 0: # Should be caught by strip() but as a safeguard
        return 0.0

    with torch.no_grad():
        outputs = causal_lm_model(**inputs, labels=input_ids.clone()) # Pass labels for loss calculation
        # The 'loss' here is the average negative log-likelihood.
        # Total log-likelihood = -loss * number_of_tokens_in_loss_calculation
        # The number of tokens contributing to the loss is typically sequence_length - 1 for Causal LMs,
        # or more accurately, the number of non-ignored tokens (non-pad, non-label=-100).
        
        # For a more direct sum of log probabilities:
        logits = outputs.logits
        total_log_prob = 0.0
        
        # Iterate through the sequence to calculate log P(token_i | previous_tokens)
        # Logits shape: (batch_size, sequence_length, vocab_size)
        # Input_ids shape: (batch_size, sequence_length)
        for i in range(input_ids.shape[1] - 1): # Predict token i+1 from tokens 0 to i
            current_logits = logits[:, i, :] # Logits for predicting token at position i+1
            target_token_id = input_ids[:, i+1] # The actual token at position i+1
            
            # Get log probabilities (log_softmax)
            log_probs = torch.log_softmax(current_logits, dim=-1)
            
            # Get the log probability of the actual target token
            # Use gather to pick the log_prob of the target_token_id
            token_log_prob = log_probs.gather(-1, target_token_id.unsqueeze(-1)).squeeze(-1)
            
            # Consider attention mask if padding was involved, though for a single sequence
            # and labels=input_ids, the loss calculation handles this.
            # Here, we sum manually, so we should only sum for actual tokens.
            if inputs.attention_mask[:, i+1].item() == 1: # Only if not a pad token
                 total_log_prob += token_log_prob.item()
        
        # Alternative using model's loss (average negative log likelihood)
        # if outputs.loss is not None:
        #     # Count non-ignored tokens (labels are input_ids, so all are considered unless tokenizer sets -100)
        #     num_tokens_in_loss = (input_ids != tokenizer.pad_token_id).sum().item()
        #     if input_ids.shape[1] > 1: # Causal LM loss often ignores first token if labels shift
        #         num_tokens_in_loss = max(0, num_tokens_in_loss -1) # Adjust if model shifts labels internally for loss
            
        #     if num_tokens_in_loss > 0 :
        #         total_log_likelihood = -outputs.loss.item() * num_tokens_in_loss
        #         return total_log_likelihood
        #     else:
        #         return 0.0 # No tokens to calculate loss over
        # else:
        #     print("Warning: Model did not return loss. Manual log_prob sum will be used.")
        #     return total_log_prob # from manual calculation above

        return total_log_prob

    return -float('inf') # Should not be reached if logic is correct

def evaluate_llm_response(tokenizer, embedding_model, causal_lm_model, generated_response: str, target_response: str) -> dict:
    """
    Evaluates the LLM's generated response against a target response using
    multiple metrics.

    Args:
        generated_response: The text generated by the LLM.
        target_response: The reference target text.

    Returns:
        A dictionary containing the scores.
    """
    scores = {
        "token_match_percentage": calculate_token_match_percentage(tokenizer, embedding_model, causal_lm_model, generated_response, target_response),
        "embedding_cosine_similarity": calculate_embedding_cosine_similarity(tokenizer, embedding_model, causal_lm_model, generated_response, target_response),
        "log_likelihood": calculate_log_likelihood(tokenizer, embedding_model, causal_lm_model, target_response) - calculate_log_likelihood(tokenizer, embedding_model, causal_lm_model, generated_response)
    }
    return scores

if __name__ == "__main__":
    try:
        # Initialize models and tokenizer globally
        # IMPORTANT: Change MODEL_NAME at the top if "meta-llama/Llama-3.2-1B-Instruct" is not suitable/available
        # Using a very small model like "gpt2" for quick testing of the script structure:
        # current_model_for_test = "gpt2" 
        current_model_for_eval = MODEL_NAME # Use the one defined at the top
        dataset_name = 'bioasq'
        print(f"--- Attempting to load model: {current_model_for_eval} ---")
        print("--- This may take a while and require significant resources. ---")
        tokenizer_eval, embedding_model_eval, causal_lm_model_eval = setup_models_and_tokenizer(current_model_for_eval)
        df, df1 = load_dataset(dataset_name)
        
        num_questions_to_run = 100
        NUM_RETRIEVED_DOCS = 10
        SEED = 42
        
        print(f"Running experiments for {num_questions_to_run} questions...")
        # Initialize Accelerator ONCE
        accelerator_main = Accelerator(mixed_precision="fp16")

        if accelerator_main.is_main_process:
            print(f"Main Script: Loading model...")
        model_path = "meta-llama/Llama-3.2-3B-Instruct" # Example, ensure this path is correct
        model_cpu = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16
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
            
        all_scores = {}
        for i in tqdm(range(num_questions_to_run), desc="Processing Questions", disable=not accelerator_main.is_main_process):
            query = df.question[i]
            answer = df.answer[i]
            if accelerator_main.is_main_process:
                print(f"\n--- Question {i+1}/{num_questions_to_run}: {query[:60]}... ---")

            docs=df1.passage[df.relevant_passage_ids[i][:NUM_RETRIEVED_DOCS]].tolist()
            
            target_response = llm_generate_response(context_str="\n\n".join(docs), tokenizer=tokenizer, device=accelerator_main.device, model=prepared_model, query=query, accelerator=accelerator_main)
            
            # LLM response evaluation block   
            print('Ground truth response: ', answer)
            print('Generated response: ', target_response)
            scores_eval = evaluate_llm_response(tokenizer_eval, embedding_model_eval, causal_lm_model_eval, target_response, answer)
            for metric, score in scores_eval.items():
                print(f"{metric}: {score:.4f}")
                try:
                    all_scores[metric].append(score)
                except:
                    all_scores[metric] = [score]
            print("-" * 40)
            
            
        print("\n\n--- Mean LLM eval metrics for the whole data ---')
        for metric in all_scores:
            print(metric, np.mean(all_scores[metric]))

    except Exception as e:
        print(f"\nAn error occurred during the execution: {e}")
        print("Please check the MODEL_NAME, your internet connection, and Hugging Face Hub access/token if applicable.")
        import traceback
        traceback.print_exc() 
