"""
TextilBot — Interface Streamlit
app.py : Chat RAG avec historique de session multi-tours
"""

import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq

from session_history import (
    init_session,
    add_user_message,
    add_bot_message,
    build_conversation_context,
    get_session_stats,
    export_history_txt,
    export_history_json,
    clear_history,
)

# ─────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="TextilBot",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --navy:   #1B2A4A;
        --gold:   #C9A84C;
        --light:  #F5F7FA;
        --border: #D6DCE8;
        --user-bg:#E8F0FE;
        --bot-bg: #FFFFFF;
    }
    .stApp { background-color: var(--light); }
    [data-testid="stSidebar"] { background-color: var(--navy); color: white; }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--gold) !important;
        border-bottom: 1px solid var(--gold);
        padding-bottom: 6px;
    }
    .msg-user {
        background: var(--user-bg);
        border-left: 4px solid #4A90D9;
        border-radius: 0 12px 12px 0;
        padding: 12px 16px;
        margin: 10px 0;
        max-width: 85%;
        margin-left: auto;
    }
    .msg-bot {
        background: var(--bot-bg);
        border-left: 4px solid var(--gold);
        border-radius: 0 12px 12px 0;
        padding: 12px 16px;
        margin: 10px 0;
        max-width: 85%;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .msg-label { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
                 text-transform: uppercase; margin-bottom: 4px; color: #888; }
    .msg-label.user { color: #4A90D9; }
    .msg-label.bot  { color: var(--gold); }
    .msg-time { font-size: 0.68rem; color: #bbb; float: right; margin-top: -18px; }
    .textilbot-header {
        display: flex; align-items: center; gap: 12px;
        padding: 12px 0 20px 0;
        border-bottom: 2px solid var(--gold);
        margin-bottom: 24px;
    }
    .textilbot-header h1 { margin: 0; font-size: 1.8rem; color: var(--navy); font-weight: 800; }
    .textilbot-header span.sub { font-size: 0.85rem; color: #888; }
    .source-badge {
        display: inline-block; background: #EEF2FF; color: var(--navy);
        border: 1px solid var(--border); border-radius: 20px;
        padding: 2px 10px; font-size: 0.73rem; margin: 3px 3px 0 0; font-weight: 600;
    }
    .stat-box {
        background: rgba(255,255,255,0.1); border-radius: 8px;
        padding: 8px 12px; margin: 4px 0; font-size: 0.82rem;
    }
    .stButton > button {
        background-color: var(--navy) !important; color: white !important;
        border-radius: 8px !important; font-weight: 600 !important;
        border: none !important; width: 100%;
    }
    .stButton > button:hover { background-color: var(--gold) !important; color: var(--navy) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# INIT SESSION
# ─────────────────────────────────────────────
init_session()


# ─────────────────────────────────────────────
# RESSOURCES (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Chargement du modèle d'embeddings...")
def load_embedding_model():
    return SentenceTransformer("BAAI/bge-m3")

@st.cache_resource(show_spinner="⏳ Connexion à Qdrant Cloud...")
def load_qdrant_client():
    return QdrantClient(
        url=st.secrets["QDRANT_URL"],
        api_key=st.secrets["QDRANT_API_KEY"],
    )

@st.cache_resource
def load_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])


# ─────────────────────────────────────────────
# FONCTIONS RAG
# ─────────────────────────────────────────────
COLLECTION_NAME = "textilbot"

def retrieve(question: str, top_k: int = 5) -> list[dict]:
    embedding_model = load_embedding_model()
    qdrant = load_qdrant_client()
    query_vector = embedding_model.encode(question, normalize_embeddings=True).tolist()
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
    )
    return [
        {"text": hit.payload["text"], "source": hit.payload["source"], "score": round(hit.score, 3)}
        for hit in results
    ]


def generate_answer(question: str, chunks: list[dict], conversation_history: list[dict]) -> str:
    """Génère une réponse en tenant compte de l'historique multi-tours."""
    groq_client = load_groq_client()

    context = ""
    for chunk in chunks:
        context += f"[Source : {chunk['source']}]\n{chunk['text']}\n\n"

    system_prompt = (
        "Tu es TextilBot, un assistant expert en conformité réglementaire textile. "
        "Tu réponds aux questions sur les normes textiles (GOTS, UE 1007/2011, ISO-1833, REACH) "
        "en te basant uniquement sur les documents fournis.\n\n"
        "Règles :\n"
        "- Réponds en français\n"
        "- Cite toujours la source (GOTS, UE 1007/2011, ISO-1833)\n"
        "- Si l'information n'est pas dans le contexte, dis-le clairement\n"
        "- Si la question fait référence à la conversation précédente, utilise ce contexte\n"
        "- Sois précis et professionnel"
    )

    # Construction des messages : system + historique + nouvelle question avec contexte
    messages = [{"role": "system", "content": system_prompt}]

    # Historique des tours précédents (sans le dernier message user qui sera ajouté ci-dessous)
    messages.extend(conversation_history[:-1])

    # Dernier message : question + contexte documentaire
    user_prompt = (
        f"Contexte documentaire :\n{context}\n"
        f"Question : {question}\n\n"
        "Réponds de manière précise en citant les sources."
    )
    messages.append({"role": "user", "content": user_prompt})

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
    )
    return response.choices[0].message.content


def ask_textilbot(question: str, top_k: int = 5) -> dict:
    """Pipeline RAG complet avec mémoire de conversation."""
    chunks = retrieve(question, top_k=top_k)
    history = build_conversation_context(max_turns=5)
    answer = generate_answer(question, chunks, history)
    return {"answer": answer, "sources": chunks}


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧵 TextilBot")
    st.markdown("Assistant IA — **conformité réglementaire textile**")
    st.divider()

    st.markdown("### 📚 Normes couvertes")
    st.markdown("""
- 🌿 **GOTS 7.0** — Certification bio
- 🏷️ **UE 1007/2011** — Étiquetage fibres
- ⚗️ **REACH** — Substances chimiques
- 🔬 **ISO 1833** — Analyse fibres
    """)
    st.divider()

    # ── Statistiques de session ──
    st.markdown("### 📊 Session en cours")
    stats = get_session_stats()
    st.markdown(f"""
<div class="stat-box">🕐 Démarrée : {stats['session_start']}</div>
<div class="stat-box">❓ Questions posées : <strong>{stats['total_questions']}</strong></div>
<div class="stat-box">📄 Sources consultées : <strong>{len(stats['sources_consulted'])}</strong></div>
    """, unsafe_allow_html=True)
    if stats["sources_consulted"]:
        st.caption("Normes utilisées : " + " · ".join(stats["sources_consulted"]))
    st.divider()

    # ── Paramètres ──
    st.markdown("### ⚙️ Paramètres")
    top_k = st.slider("Chunks récupérés (top-k)", 3, 10, 5)
    show_sources = st.toggle("Afficher les sources", value=True)
    max_turns = st.slider("Mémoire (tours)", 1, 10, 5,
                          help="Nombre d'échanges précédents transmis au LLM")
    st.divider()

    # ── Export historique ──
    st.markdown("### 💾 Exporter l'historique")
    if st.session_state.messages:
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📄 TXT",
                data=export_history_txt(),
                file_name=f"textilbot_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
                          if False else "textilbot_historique.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                label="📦 JSON",
                data=export_history_json(),
                file_name="textilbot_historique.json",
                mime="application/json",
                use_container_width=True,
            )
    else:
        st.caption("Aucun message à exporter.")
    st.divider()

    # ── Actions ──
    if st.button("🗑️ Effacer la conversation"):
        clear_history()
        st.rerun()

    st.markdown("### 💡 Questions exemples")
    example_questions = [
        "Exigences minimales GOTS ?",
        "Que doit contenir l'étiquette textile UE ?",
        "Méthode d'analyse ISO 1833 pour mélanges ?",
        "Substances chimiques interdites dans les textiles ?",
    ]
    for q in example_questions:
        if st.button(q, key=f"ex_{q}"):
            st.session_state._quick_question = q
            st.rerun()

# Import datetime ici pour l'export
from datetime import datetime


# ─────────────────────────────────────────────
# ZONE PRINCIPALE
# ─────────────────────────────────────────────
st.markdown("""
<div class="textilbot-header">
    <span style="font-size:2.4rem">🧵</span>
    <div>
        <h1>TextilBot</h1>
        <span class="sub">Conformité réglementaire textile — GOTS · UE 1007/2011 · ISO 1833 · REACH</span>
    </div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.messages:
    st.info(
        "👋 Bonjour ! Je suis **TextilBot**. Posez-moi vos questions sur les normes GOTS, "
        "l'étiquetage UE, les substances REACH ou l'analyse ISO 1833. "
        "Je me souviens du contexte de notre conversation !",
        icon="🧵",
    )

# ── Affichage historique ──
for msg in st.session_state.messages:
    time_str = msg.get("timestamp", "")
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="msg-user">
            <div class="msg-label user">Vous <span class="msg-time">{time_str}</span></div>
            {msg["content"]}
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="msg-bot">
            <div class="msg-label bot">🧵 TextilBot <span class="msg-time">{time_str}</span></div>
            {msg["content"]}
        </div>""", unsafe_allow_html=True)
        if show_sources and msg.get("sources"):
            sources_html = "".join(
                f'<span class="source-badge">📄 {s["source"]} — {s["score"]}</span>'
                for s in msg["sources"]
            )
            st.markdown(f"<div style='margin-top:6px'>{sources_html}</div>", unsafe_allow_html=True)

st.divider()

# ── Saisie ──
col_input, col_btn = st.columns([5, 1])
with col_input:
    user_input = st.text_input(
        label="Question",
        placeholder="Ex : Et pour les mélanges laine-polyester, quelle méthode ISO ?",
        key=f"chat_input_{st.session_state.input_key}",
        label_visibility="collapsed",
    )
with col_btn:
    send = st.button("Envoyer ➤")

# Question rapide depuis sidebar
if hasattr(st.session_state, "_quick_question"):
    user_input = st.session_state._quick_question
    del st.session_state._quick_question
    send = True

# ── Pipeline ──
if send and user_input.strip():
    add_user_message(user_input)

    with st.spinner("🔍 Recherche dans les documents..."):
        try:
            result = ask_textilbot(user_input, top_k=top_k)
            add_bot_message(result["answer"], result["sources"])
        except Exception as e:
            add_bot_message(f"❌ Erreur : {e}", [])

    st.session_state.input_key += 1
    st.rerun()
