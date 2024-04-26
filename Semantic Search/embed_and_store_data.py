"""
- Prepare the text to embed for each reccord of your dataset.
    - Create the reccord.
        - Clean the text.
        - Concatenate fields.
- Choose a Sentence Embedding Model.
- Embed the text generated in the previous step for each reccord.
- Store the embeddings in a vector database (i.e. elasticsearch).
"""
import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from elasticsearch import Elasticsearch
from tqdm import tqdm

def embed_data():

    # Load preprocessed dataset
    imdb_data = pd.read_csv("processed_imdb_data.csv")

    # Initialize Huggingface Sentence Transformers model
    model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')

    # Encode text data into embeddings

    embeddings = []
    with tqdm(total=len(imdb_data)) as pbar:
        for text in imdb_data['Concatenated_Text'].tolist():
            embeddings.append(model.encode([text])[0])
            pbar.update(1)

    return model,embeddings

            
def store_data(es, embeddings):
        
    # es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

    # Define index mapping
    index_mapping = {
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "dense_vector",
                    "dims": len(embeddings[0])  # Dimensionality of embeddings
                }
            }
        }
    }

    # Create Elasticsearch index
    index_name = "imdb_embeddings"
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=index_mapping)

    # Store embeddings in Elasticsearch
    for idx, embedding in enumerate(embeddings):
        document_id = idx  # Assuming document ID corresponds to the index of the DataFrame
        es.index(index=index_name, id=document_id, body={"embedding": embedding.tolist()})


    print("Embeddings stored in the vector database successfully.")

    return es, index_name
