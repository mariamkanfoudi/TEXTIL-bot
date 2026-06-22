"""
session_history.py — Gestion de l'historique de conversation TextilBot
Stocke, formate et exporte la mémoire de session Streamlit.
"""

from datetime import datetime
import json
import streamlit as st


# ─────────────────────────────────────────────
# INITIALISATION
# ─────────────────────────────────────────────

def init_session():
    """Initialise toutes les clés de session nécessaires."""
    if "messages" not in st.session_state:
        st.session_state.messages = []          # historique complet
    if "session_start" not in st.session_state:
        st.session_state.session_start = datetime.now().strftime("%d/%m/%Y %H:%M")
    if "question_count" not in st.session_state:
        st.session_state.question_count = 0
    if "input_key" not in st.session_state:
        st.session_state.input_key = 0


# ─────────────────────────────────────────────
# AJOUT DE MESSAGES
# ─────────────────────────────────────────────

def add_user_message(content: str):
    """Ajoute un message utilisateur à l'historique."""
    st.session_state.messages.append({
        "role":      "user",
        "content":   content,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    })
    st.session_state.question_count += 1


def add_bot_message(content: str, sources: list[dict] = None):
    """Ajoute une réponse TextilBot à l'historique."""
    st.session_state.messages.append({
        "role":      "bot",
        "content":   content,
        "sources":   sources or [],
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    })


# ─────────────────────────────────────────────
# CONSTRUCTION DU CONTEXTE MULTI-TOURS
# ─────────────────────────────────────────────

def build_conversation_context(max_turns: int = 5) -> list[dict]:
    """
    Retourne les N derniers échanges formatés pour Groq (multi-turn).
    Chaque échange = 1 message user + 1 message assistant.
    
    max_turns : nombre de tours (paires question/réponse) à conserver
    """
    messages = st.session_state.messages

    # Ne garder que les N derniers tours (chaque tour = 2 messages)
    recent = messages[-(max_turns * 2):]

    history = []
    for msg in recent:
        if msg["role"] == "user":
            history.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "bot":
            history.append({"role": "assistant", "content": msg["content"]})

    return history


# ─────────────────────────────────────────────
# STATISTIQUES DE SESSION
# ─────────────────────────────────────────────

def get_session_stats() -> dict:
    """Retourne les statistiques de la session en cours."""
    messages = st.session_state.messages
    sources_used = set()

    for msg in messages:
        for src in msg.get("sources", []):
            sources_used.add(src.get("source", ""))

    return {
        "total_questions":  st.session_state.question_count,
        "total_messages":   len(messages),
        "session_start":    st.session_state.get("session_start", "—"),
        "sources_consulted": list(sources_used),
    }


# ─────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────

def export_history_txt() -> str:
    """Exporte l'historique complet en texte brut."""
    lines = [
        "═══════════════════════════════════════",
        "       HISTORIQUE — TextilBot",
        f"  Session du {st.session_state.get('session_start', '—')}",
        "═══════════════════════════════════════\n",
    ]
    for msg in st.session_state.messages:
        role  = "Vous" if msg["role"] == "user" else "TextilBot"
        time  = msg.get("timestamp", "")
        lines.append(f"[{time}] {role} :")
        lines.append(msg["content"])
        if msg.get("sources"):
            src_names = ", ".join(s["source"] for s in msg["sources"])
            lines.append(f"  📄 Sources : {src_names}")
        lines.append("")
    return "\n".join(lines)


def export_history_json() -> str:
    """Exporte l'historique complet en JSON."""
    payload = {
        "session_start": st.session_state.get("session_start", ""),
        "stats":         get_session_stats(),
        "messages":      st.session_state.messages,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────
# EFFACEMENT
# ─────────────────────────────────────────────

def clear_history():
    """Réinitialise complètement la session."""
    st.session_state.messages       = []
    st.session_state.question_count = 0
    st.session_state.session_start  = datetime.now().strftime("%d/%m/%Y %H:%M")
    st.session_state.input_key     += 1
