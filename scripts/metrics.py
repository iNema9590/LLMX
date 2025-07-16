import numpy as np


def softmax(x): 
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def top_k_removal( original_logits: list, removal_logits: list, 
        top_k: int, target_token_id: int, aggregation: str='mean'):
    if len(removal_logits) != top_k :
        raise ValueError(f'Expected {top_k} removal logits, got {len(removal_logits)}')
    
    original_probs = softmax(original_logits)

    # compute metrics for each removal
    prob_changes = []
    rank_changes = []
    kl_divergences = []

    for elem in removal_logits: 
        removal_probs = softmax(elem)

        # Prob Changes
        prob_change = original_probs[target_token_id] - removal_probs[target_token_id]
        prob_changes.append(prob_change)

        #Rank Changes
        original_rank = len(np.where(original_probs > original_probs[target_token_id])[0])
        removal_rank = len(np.where(removal_probs > removal_probs[target_token_id])[0])
        rank_changes.append(original_rank - removal_rank)

        # KL divergence
        kl_div = np.sum(removal_probs * np.log(removal_probs / original_probs))
        kl_divergences.append(kl_div)

    if aggregation == 'mean': 
        agg_prob_change = np.mean(prob_changes)
        



    