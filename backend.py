import os 
import requests
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
import time
import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
import cohere
from sentence_transformers import SentenceTransformer



load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]-%(message)s"
)

logger=logging.getLogger(__name__)

try:
    genai.configure(api_key=os.getenv('API_Key'))
    qdrant=QdrantClient(url=os.getenv('qdrant_url'),api_key=os.getenv('qdrant_api_key'),timeout=60)
   
    gemini=genai.GenerativeModel("gemini-1.5-pro")
    co=cohere.Client(api_key=os.getenv("CHOHER_API_KEY"))
    collection_name='prac__rag_docs'
    embedder=SentenceTransformer('all-MiniLM-L6-v2')
    if not qdrant.collection_exists(collection_name):
        qdrant.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=384,distance=models.Distance.COSINE)
        )
        logger.info("qdrant collection created succesfully")
        print('collections created succesfully')
        
    else:
        logger.info('collectio already exists os skipp')
        print('collection already exists')
except Exception as e:
    logger.error(f'Error in setup{e}')
    raise
def read_pdf_return_emb(pdf_path):
    try:
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " "]
        )
        
        chunks = []
        for doc in docs:
            print(doc)
            splits = text_splitter.split_text(doc.page_content)
            chunks.extend(splits)  # use extend to flatten all chunks

        points = []
        for i, chunk in enumerate(chunks):
            embed = embedder.encode(chunk).tolist()
            points.append(
                models.PointStruct(
                    id=i,
                    vector=embed,
                    payload={'text': chunk}
                )
            )

        qdrant.upsert(
            collection_name=collection_name,
            points=points
        )
        logger.info(f"PDF {pdf_path} processed and uploaded successfully.")

    except Exception as e:
        logger.error(f"Error reading or embedding the PDF: {e}")
        raise


def retrieve(query,top_k=10):
    try:
        query_vec=embedder.encode(query).tolist()
        results=qdrant.search(
            collection_name=collection_name,
            query_vector=query_vec,
            limit=top_k
        )
        logger.info(f"Retrived {len(results)} results for query :{query}")
        return results
    except Exception as e:
        logger.error(f"error in retruval :{e}")
        return []
    
    
def answer_query(query):
    try:
        retrieved = retrieve(query, top_k=10)
        if not retrieved:
            return "No relevant results found", []
        docs = [r.payload['text'] for r in retrieved]
        rerank_results = co.rerank(
            query=query,
            documents=docs,
            top_n=3,
            model="rerank-english-v3.0"
        )
        reranked_docs = [docs[r.index] for r in rerank_results.results]
        context = "\n".join(reranked_docs)
        prompt = f"""
        Use the following context to answer the query.
        Context:
        {context}
        Query: {query}
        Answer with citations like [1], [2].
        """
        response = genai.GenerativeModel('gemini-1.5-flash').generate_content(prompt)
        logger.info("Generated answer successfully")
        return response.text, reranked_docs
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "Error generating answer", []
st.title("RAG with Gemini + Qdrant")
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    read_pdf_return_emb("temp.pdf")
    st.success("PDF uploaded and processed")

query = st.text_input("Ask a question")
if st.button("Submit Query") and query:
    start = time.time()
    answer, retrieved_docs = answer_query(query)
    end = time.time()
    st.write("Answer:")
    st.write(answer)
    st.write(f"Response time: {end - start:.2f} seconds")

    # Optional: Estimate token usage
    total_chars = len(query) + len(" ".join(retrieved_docs)) + len(answer)
    est_tokens = total_chars // 4   # rough estimate: 1 token ≈ 4 chars
    st.write(f"Estimated tokens: {est_tokens}")