"""
    Create an Streamlit app that does the following:

    - Reads an input from the user
    - Embeds the input
    - Search the vector DB for the entries closest to the user input
    - Outputs/displays the closest entries found
"""

import pandas as pd
import streamlit as st
import clean_data as cd
import embed_and_store_data as em_st
import pickle
from elasticsearch import Elasticsearch


# Main Function
def main():

  st.title("IMDB Search using Semantic Similarity")

  # To clean the data 
  
  # imdb_data = pd.read_csv("imdb_top_1000.csv")
  # cd.clean_data(imdb_data)

  # Data embedding using Huggingface Sentence Transformer

  # model, embeddings = em_st.embed_data()

  # Storing vector embeddings using ElasticSearch in Progress

  es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

  with open('embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)


  es, index_name = em_st.store_data(es, embeddings)


  with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

 # query = "henry hill and his life in the mob covering his relationship"
 # query = "presidencies of kennedy and johnson the events of vietnam watergate and other historical events"
  user_input = st.text_input("Enter your search query:")
    
    # Wait until the user provides input
  if user_input == "":
      st.text("Waiting for input...")  # Display a message while waiting
      
  if user_input != "":
      
    st.write("Input obtained:", user_input)
  
    query_embedding = model.encode([user_input])[0]

    search_results = es.search(
      index=index_name,
      body={
          "query": {
              "script_score": {
                  "query": {"match_all": {}},
                  "script": {
                      "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                      "params": {"query_vector": query_embedding.tolist()}
                  }
              }
          }
        }
      )

    imdb = pd.read_csv("cleaned_imdb_data.csv",  usecols=['Series_Title','Overview'])
    st.subheader("Search Results:")
    for hit in search_results['hits']['hits']:
      document_id = hit['_id']
      score = hit['_score']
      print(f"Document ID: {document_id}, Score: {score}")
      print(imdb.Series_Title.get(int(document_id)), imdb.Overview.get(int(document_id)))
      message = "Title : " + imdb.Series_Title.get(int(document_id)), "Overview : " + imdb.Overview.get(int(document_id))
      st.success(message)

 
if __name__ == "__main__":
  main()
