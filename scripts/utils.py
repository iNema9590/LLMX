
import numpy as np

# def precision_at_k(ground_truth, model_scores, k=3):

#     relevance_threshold = np.median(ground_truth) # get the relevance threshold. 
#     # Rank model predictions (highest score first)
#     top_k_indices = sorted(range(len(model_scores)), key=lambda i: model_scores[i], reverse=True)[:k] # get the indexes of the highes scores
#     # Count relevant items in top-k based on ground truth, we check if the scores in ground truth are considered relevant. 
#     relevant_count = sum(1 for i in top_k_indices if ground_truth[i] >= relevance_threshold)
    
#     return relevant_count / k

def precision_at_k(ground_truth, model_scores, k = 3): 

    top_k_indices = sorted(range(len(model_scores)), key = lambda i: model_scores[i], reverse = True )[:k]
    ground_truth_relevant_indices = sorted(range(len(ground_truth)), key = lambda i: ground_truth[i], reverse = True)[:k]
    relevant_in_top_k = sum(1 for idx in top_k_indices if idx in ground_truth_relevant_indices)
    return relevant_in_top_k / k

def hit_rate_at_k(ground_truth, model_scores, k=3):
    """Different from precision at k because it only says if there is at least one relevant itm included in top k. 
    if there is then the output will be 1 (we don't care how many irrelevant items are included in top k, 
    what matters is that there is at least 1)

    Args:
        model_scores (_type_): list of predicted scores (float)
        ground_truth (_type_): list of scores (float)
        k (int, optional): top k to look for relevant docs. Defaults to 3.

    Returns:
        _type_: _description_
    """
    relevance_threshold = np.median(ground_truth) # get the relevance threshold. 
    # Get indices of top-k predicted scores
    top_k_indices = sorted(range(len(model_scores)), key=lambda i: model_scores[i], reverse=True)[:k]
    
    # Check if any of the top-k items are relevant
    for i in top_k_indices:
        if ground_truth[i] >= relevance_threshold:
            return 1  # Hit
    return 0  # Miss

def reciprocal_rank(ground_truth, model_scores):
    """ RR is a ranking metric that measures the position of the first relevant item in a ranked list. 
    If the first relevant item is at position 1 -> RR = 1
    If the first relevant item is at position 4 -> RR = 1/4 = 0.25
    If no relevant item is found -> RR = 0

    Args:
        model_scores (list): predicted scores 
        ground_truth (list): ground truth scores

    Returns:
        _type_: _description_
    """

    relevance_threshold = np.median(ground_truth) # get the relevance threshold. 

    # Sort indices by model scores (highest first)
    ranked_indices = sorted(range(len(model_scores)), key=lambda i: model_scores[i], reverse=True)
    
    # Find the rank (1-based) of the first relevant item
    for rank, idx in enumerate(ranked_indices, start=1):
        if ground_truth[idx] >= relevance_threshold:
            return 1.0 / rank  # Found the first relevant item
    return 0.0  # No relevant item found