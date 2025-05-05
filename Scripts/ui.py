import os
import streamlit as st
import pandas as pd
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

# --- Setup paths ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
index_path = os.path.join(root_dir, "embeddings", "faiss_index.index")
id_map_path = os.path.join(root_dir, "embeddings", "id_map.pkl")
data_path = os.path.join(root_dir, "Dataset", "PlacementInfo.csv")

# --- Load assets ---
df = pd.read_csv(data_path)
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index(index_path)
with open(id_map_path, "rb") as f:
    id_map = pickle.load(f)

# --- Gemini setup ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyD31LxkZ-MLGC9oOtr8jYK0qTfyr1vXyo4"
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.environ["GOOGLE_API_KEY"])

# --- Query logic ---
def search_faiss(query, k=2):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    responses = [df.iloc[i]["Answer"] for i in indices[0] if i < len(df)]
    return responses

def search_gemini(query):
    prompt = ChatPromptTemplate.from_template("Search this query relating to the placements in engineering in India, please answer in short and be very precise and clear as well as polite to user. try to give answers in bullet point in about 5-6 in number: {query}")
    chain = prompt | llm
    result = chain.invoke({"query": query})
    return result.content

# --- Streamlit UI ---
st.title("🎓 PlacementBot - College Query Assistant")
user_query = st.text_input("Ask your placement-related question:")

if user_query:
    faiss_answers = search_faiss(user_query)
    
    if faiss_answers:
        st.subheader("📚 Answer from your Dataset:")
        for ans in faiss_answers:
            st.write(f"- {ans}")
    else:
        st.info("No relevant answer found in your dataset. Searching via Gemini...")

    gemini_response = search_gemini(user_query)
    st.subheader("🌐 Gemini Response:")
    st.write(gemini_response)
