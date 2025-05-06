import requests
from elasticsearch import Elasticsearch, helpers
from concurrent.futures import ThreadPoolExecutor, as_completed

def connect_elasticsearch():
    """Connect to Elasticsearch."""
    return Elasticsearch('http://localhost:9200')

def create_index(es_client, index_name, mapping):
    """Create an Elasticsearch index."""
    if not es_client.indices.exists(index=index_name):
        es_client.indices.create(index=index_name, body=mapping)

def index_documents(es_client, index_name, documents):
    """Index documents into Elasticsearch."""
    actions = [
        {
            "_index": index_name,
            "_id": doc["id"],
            "_source": doc
        }
        for doc in documents
    ]
    helpers.bulk(es_client, actions)

def retrieve_documents(es_client, index_name, query, top_k=5):
    """Retrieve relevant documents from Elasticsearch."""
    search_query = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["title", "abstract", "authors"]
            }
        },
        "size": top_k
    }
    response = es_client.search(index=index_name, body=search_query)
    return [hit["_source"] for hit in response["hits"]["hits"]]


def generate_response(query, documents, model="llama3.3"):
    """
    Generate a response using the Ollama API for a given subset of documents.
    """
    url = "http://localhost:11434/api/generate"

    # Combine the documents into a single context
    context = "\n\n".join([f"{doc['title']}. {doc['abstract']}" for doc in documents])

    # Prepare the prompt for Llama3
    prompt = f"Learn from the whole context and answer the question. No need to refer to the context answer like you knew. \n Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    # Send the prompt to Llama3
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=data).json()
    return response["response"]

def generate_responses_parallel(query, subsets, num_workers=4):
    """
    Generate responses for multiple subsets of documents in parallel.
    """
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(generate_response, query, subset) for subset in subsets]
        responses = [future.result() for future in as_completed(futures)]
    return responses