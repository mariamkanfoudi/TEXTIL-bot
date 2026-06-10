import streamlit as st
from rag_utils import retrieve_context, generate_answer

st.set_page_config(page_title="TextilBot", page_icon="🧵")
st.title("🧵 TextilBot – Assistant conformité textile")
st.markdown("Posez une question sur les normes **GOTS**, **UE 1007/2011** ou **ISO 1833**.")

# Initialiser l'état de session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher l'historique
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Champ de saisie
question = st.chat_input("Votre question sur les textiles...")
if question:
    # Afficher la question
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Récupérer les chunks pertinents
    with st.spinner("🔍 Recherche dans la base documentaire..."):
        chunks = retrieve_context(question, top_k=5)

    # Générer la réponse
    with st.spinner("🤖 Génération de la réponse..."):
        answer = generate_answer(question, chunks)

    # Afficher la réponse
    with st.chat_message("assistant"):
        st.markdown(answer)

    # Afficher les sources (dépliable)
    with st.expander("📚 Voir les sources utilisées"):
        for i, chunk in enumerate(chunks):
            st.write(f"**Source {i+1}** : {chunk['source']} (score : {chunk['score']:.3f})")
            st.caption(chunk['text'][:500] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})