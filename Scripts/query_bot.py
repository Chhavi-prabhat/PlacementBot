# This script:

# Takes a user query

# Searches your FAISS vector store

# Returns an answer from your dataset if relevant

# Otherwise, falls back to Gemini for a generative response

import os
import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from gemini_fallbck import search_gemini
from langchain_core.messages import HumanMessage, SystemMessage

chat_history=[]

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)

data_path = os.path.join(root_dir, "Dataset", "PlacementInfo.csv")
index_path = os.path.join(root_dir, "embeddings", "faiss_index.index")
id_map_path = os.path.join(root_dir, "embeddings", "id_map.pkl")

# print(index_path)
df = pd.read_csv("../Dataset/PlacementInfo.csv")
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index(index_path)

with open(id_map_path, "rb") as f:
    id_map = pickle.load(f)

def query_bot(user_query, top_k=1, faiss_threshold=1.0):
    query_embedding = model.encode([user_query])
    distances, indices = index.search(query_embedding, top_k)
    
    # If match is good enough
    if distances[0][0] < faiss_threshold:
        matched_id = id_map[indices[0][0]]
        answer = df.iloc[matched_id]["Answer"]
        return f"[FAISS Answer]\n{answer}"
    
    # Otherwise use Gemini
    gemini_response = search_gemini(user_query)
    return f"[Gemini Answer]\n{gemini_response}"

# ----------- Example Run -----------
if __name__ == "__main__":
    while True:
        user_input = input("\nAsk a placement-related question (or type 'exit'): ")
        if user_input.lower() == "exit":
            break
        response = query_bot(user_input)
        print("\n" + response)