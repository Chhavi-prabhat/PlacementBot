# This script:

# Takes a user query

# Searches query using gemini llm using api key

# Returns an answer from gemini

import os
from langchain_google_genai import ChatGoogleGenerativeAI
# from google.generativeai import genai
from langchain.prompts import ChatPromptTemplate

os.environ["GOOGLE_API_KEY"] = "AIzaSyD31LxkZ-MLGC9oOtr8jYK0qTfyr1vXyo4"

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.environ["GOOGLE_API_KEY"])

def search_gemini(query):
    prompt=ChatPromptTemplate.from_template("You are an helpful ai assistant, be nice to user . Search this query relating to the placements in engineering in India  {query}. Be precise and give answer in 3-5 lines")
    chain=prompt | llm
    result=chain.invoke({"query": query})
    return result.content

# text="Which company offers highest package?"
# res=search_gemini(text)
# print(res)