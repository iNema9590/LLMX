import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import re
from tqdm import tqdm
# Load dataset
data=pd.read_csv('data/recipes.csv')
features_drop = ['recipe_url', 'input_db_index', 'food_kg_locator', 'food_com_unique_id', 'submit_date', 'last_changed_date', 'author_id', 'rating', 'recipe_id', 'serves', 'units' ]
data.drop(features_drop, axis = 1, inplace = True)

def get_ingredients_list(ingredients:str) -> list: 
    """Clean the ingredients string to get a list. Each element of a list is a specific ingredient with the quantity

    Args:
        ingredients (str): string of ingredients

    Returns:
        list: list of ingredients (each elem in the lsit is an ingredient.)
    """
    sample_list = ingredients.split(')')
    clean_sample_list = ''
    pattern = r'[^A-Za-z0-9/ -]'
    for elem in sample_list : 
        cleaned_elem = re.sub(pattern, ' ', elem)
        cleaned_elem = re.sub(r'\s+', ' ', cleaned_elem).strip(' ')
        if len(cleaned_elem) > 1: 
            clean_sample_list=clean_sample_list+" "+cleaned_elem

    return clean_sample_list

def get_directions_list(directions:str): 
    """Clean the directions string to get a list of directions

    Args:
        directions (str): badly formated directions string

    Returns:
        list: contains the different directions to follow for a given recipe. 
    """
    return directions.strip('{'""'}').replace('","', ' ')

data['new_ingredients'] = ''
data['new_directions'] = ''
data.new_ingredients = data.ingredients.apply(lambda x : get_ingredients_list(x))
data.new_directions = data.directions.apply(lambda x : get_directions_list(x))
data.drop(['ingredients', 'directions'], axis = 1, inplace = True)
data.rename(columns = {'new_ingredients': 'ingredients', 'new_directions' : 'directions'}, inplace = True)
data["text"] =data["title"] +". Ingredients:" + data["ingredients"] + ". Instructions:" +data["directions"]
documents = data["text"].tolist()
# Load a local embedding model (e.g., sentence-transformers)
# model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
# embeddings = model.encode(documents, convert_to_numpy=True, show_progress_bar=True)


#Generate nomic embeddings
import requests
url = "http://localhost:11434/api/embeddings"
embeddings=[]
for i in tqdm(documents):
    data = {
        "model": "nomic-embed-text",
        "prompt": i
    }

    embeddings.append(requests.post(url, json=data).json()['embedding'])

embeddings=np.array(embeddings)

# Save documents separately
# with open("data/documents.pkl", "wb") as f:
#     pickle.dump(documents, f)

# Save embeddings separately
with open("data/embeddings_nomic.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print("Embeddings of nomic saved successfully.")
