# This script:

# Takes dataset

# Embeddes the token

# Store it in faiss (vector db)

import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
import faiss
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

target_dir = os.path.join(root_dir, "Dataset", "PlacementInfo.csv")
embedding_dir = os.path.join(root_dir, "embeddings")
index_path = os.path.join(embedding_dir, "faiss_index.index")
id_map_path = os.path.join(embedding_dir, "id_map.pkl")

df = pd.read_csv(target_dir)
df['id'] = df.index  # Ensure unique ID column

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['Question'].tolist(), show_progress_bar=True)
# print(embeddings)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Ensure the embeddings directory exists
os.makedirs(embedding_dir, exist_ok=True)

# Save index and ID map
faiss.write_index(index, index_path)
with open(id_map_path, "wb") as f:
    pickle.dump(df['id'].tolist(), f)