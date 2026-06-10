import os
import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq

# Chargement des secrets Streamlit
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
COLLECTION_NAME = st.secrets["COLLECTION_NAME"]

# Initialisation des clients (avec mise en cache)
@st.cache_resource
def init_qdrant():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

@st.cache_resource
def init_embedding_model():
    # Modèle multilingue – dimension 1024
    return SentenceTransformer("BAAI/bge-m3")

@st.cache_resource
def init_groq():
    return Groq(api_key=GROQ_API_KEY)

# Fonction de recherche vectorielle
def retrieve_context(query: str, top_k: int = 5):
    client = init_qdrant()
    encoder = init_embedding_model()

    # Encoder la question (normalisé)
    query_vector = encoder.encode(query, normalize_embeddings=True)

    # Recherche dans Qdrant
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector.tolist(),
        limit=top_k,
        with_payload=True
    )

    chunks = []
    for hit in results:
        chunks.append({
            "text": hit.payload["text"],
            "source": hit.payload["source"],
            "score": hit.score
        })
    return chunks

# Fonction d'appel au LLM (Groq)
def generate_answer(question: str, context_chunks: list) -> str:
    groq_client = init_groq()

    # Construction du contexte
    context_str = ""
    for chunk in context_chunks:
        context_str += f"[Source: {chunk['source']}]\n{chunk['text']}\n\n"

    system_prompt = """Tu es TextilBot, un assistant expert en conformité réglementaire textile.
Tu réponds aux questions sur les normes textiles (GOTS, réglementation UE, ISO) en te basant
uniquement sur les documents fournis.

Règles :
- Réponds en français
- Cite toujours la source (GOTS, UE 1007/2011, ISO-1833)
- Si l'information n'est pas dans le contexte, dis-le clairement
- Sois précis et professionnel"""

    user_prompt = f"""Contexte documentaire :
{context_str}

Question : {question}

Réponds de manière précise en citant les sources."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=1024
    )
    return response.choices[0].message.content