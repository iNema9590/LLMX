import numpy as np
from tqdm import tqdm
from itertools import combinations
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils import generate_response, generate_responses_parallel


def compute_similarity(text1, text2):
    """Compute cosine similarity between two texts."""
   

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([text1, text2])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

def compute_exact_shapley_values(query, documents, num_workers=4):
    """
    Compute exact Shapley values for the retrieved documents with parallel LLM calls.
    """
    num_documents = len(documents)
    shapley_values = np.zeros(num_documents)

    # Generate all possible subsets of documents
    all_subsets = []
    for subset_size in range(num_documents + 1):
        all_subsets.extend(combinations(range(num_documents), subset_size))

    # Iterate over each document
    for i in tqdm(range(num_documents)):
        contributions = []

        # Prepare subsets with and without the current document
        subsets_with = []
        subsets_without = []
        for subset in all_subsets:
            if i not in subset:
                subsets_with.append([documents[j] for j in subset] + [documents[i]])
                subsets_without.append([documents[j] for j in subset])

        # Generate responses in parallel for subsets with and without the document
        responses_with = generate_responses_parallel(query, subsets_with, num_workers=num_workers)
        responses_without = generate_responses_parallel(query, subsets_without, num_workers=num_workers)

        # Compute the marginal contribution for each subset
        for response_with, response_without in zip(responses_with, responses_without):
            similarity = compute_similarity(response_with, response_without)
            contributions.append(similarity)

        # Average the contributions to get the Shapley value
        shapley_values[i] = np.mean(contributions)

    return shapley_values


def visualize_shapley_values(shapley_values, documents):
    """Visualize the Shapley values for the retrieved documents."""
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(shapley_values)), shapley_values, tick_label=[f"Document {i+1}" for i in range(len(shapley_values))])
    plt.xlabel("Documents")
    plt.ylabel("Shapley Value")
    plt.title("DataSHAP Values for Retrieved Documents")
    plt.show()