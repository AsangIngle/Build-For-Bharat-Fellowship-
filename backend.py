import os
import time
import requests
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import cohere
import google.generativeai as genai

st.set_page_config(page_title="Project Samarth - Gov Data Q&A", layout="wide")

# Load API keys
QDRANT_URL = os.getenv("qdrant_url_2")
QDRANT_API = os.getenv("qdrant_api_2")
COHERE_API = os.getenv("CHOHER_API_KEY").strip()
GENAI_API = os.getenv("API_Key").strip()
GOV_API = os.getenv("gov_api").strip()

# Initialize clients
genai.configure(api_key=GENAI_API)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API, timeout=60)
co = cohere.Client(api_key=COHERE_API)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
COLLECTION = "gov_data_collection"

if not qdrant.collection_exists(COLLECTION):
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
    )

@st.cache_data
def fetch_data_from_api(resource_id, api_key=GOV_API, limit=5000):
    offset, all_records = 0, []
    while True:
        url = f"https://api.data.gov.in/resource/{resource_id}?api-key={api_key}&format=json&limit={limit}&offset={offset}"
        response = requests.get(url)
        response.raise_for_status()
        records = response.json().get("records", [])
        if not records:
            break
        all_records.extend(records)
        offset += limit
        time.sleep(0.5)
    return pd.DataFrame(all_records)

def embed_and_store(df, dataset_name):
    records = []
    for idx, row in df.iterrows():
        text = " ".join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col]) and str(row[col]).strip()])
        if not text:
            continue
        embedding = embedder.encode(text).tolist()
        records.append(models.PointStruct(
            id=int(idx),
            vector=embedding,
            payload={"text": text, "dataset": dataset_name}
        ))
    if records:
        qdrant.upsert(collection_name=COLLECTION, points=records)

def semantic_search(query, top_k=10):
    query_vector = embedder.encode(query).tolist()
    results = qdrant.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True
    )
    return results

def generate_answer(context, query):
    prompt = f"""
You are an intelligent assistant analyzing Indian government datasets.
Question: {query}

Relevant data context:
{context}

Please synthesize a factual, data-driven answer and cite the dataset sources clearly.
"""
    try:
        model = genai.GenerativeModel("models/gemini-2.5-pro-preview-03-25")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            st.warning("Gemini quota exceeded. Switching to Cohere as fallback.")
            try:
                response = co.generate(model='command-xlarge-nightly', prompt=prompt, max_tokens=300)
                return response.text.strip()
            except:
                return "Both Gemini and Cohere failed due to API limits."
        else:
            return "Could not generate a response due to an API error."

# --- Streamlit UI ---
st.title("Project Samarth — Intelligent Q&A on Government Data")
st.markdown("Ask complex questions using live data from data.gov.in")

# Sidebar: dataset setup
st.sidebar.header("Data Source Setup")
resource_id = st.sidebar.text_input("Resource ID", "14613c4e-5ab0-4705-b440-e4e49ae345de")
dataset_name = st.sidebar.text_input("Dataset Name", "Budget and Financial Data")

if st.sidebar.button("Load & Embed Dataset"):
    df = fetch_data_from_api(resource_id)
    if not df.empty:
        embed_and_store(df, dataset_name)
        st.success(f"Loaded and stored {len(df)} records from {dataset_name}")
    else:
        st.warning("No records found or invalid dataset.")

# Main UI: question input
st.divider()
st.subheader("Ask Your Question")
query = st.text_area("Enter your question:")

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a question first.")
    else:
        search_results = semantic_search(query, top_k=10)
        context = "\n\n".join([r.payload.get('text', '') for r in search_results if 'text' in r.payload and r.payload['text']])
        if not context.strip():
            st.warning("No relevant data found to answer your query.")
        else:
            answer = generate_answer(context, query)
            st.markdown("**Answer:**")
            st.write(answer)
            st.markdown("**Sources:**")
            for r in search_results:
                st.markdown(f"- {r.payload.get('dataset', 'Unknown')}")
