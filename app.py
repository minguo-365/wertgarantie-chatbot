import streamlit as st
import os
import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import re
import requests

st.set_page_config(page_title="🤖 Willkommen", layout="wide")

client = OpenAI(api_key=st.secrets["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")

@st.cache_resource
def init_vector_store():
    with open("wertgarantie.txt", "r", encoding="utf-8") as f:
        text = f.read()
    chunks = [chunk.strip() for chunk in text.split("\n\n") if len(chunk.strip()) > 50]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return model, chunks, index, embeddings

model, chunks, index, _ = init_vector_store()

# Suche relevante Textabschnitte basierend auf Benutzeranfrage
def get_relevant_chunks(query, k=3):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), k)
    return [(chunks[i], i) for i in I[0]]

st.title("🤖 Willkommen")
st.markdown("**Ich bin Ihr digitaler Assistent.**")

if st.button("🩹  Verlauf löschen"):
    st.session_state.chat_history = []
    st.rerun()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialisiere Sub-Button-SessionStates
link_keys = ["show_link_smartphone", "show_link_notebook", "show_link_kamera", "show_link_tv", "show_link_werkstatt", "show_link_haendler"]
for key in link_keys:
    if key not in st.session_state:
        st.session_state[key] = False

if "show_sub_buttons" not in st.session_state:
    st.session_state.show_sub_buttons = False

USER_AVATAR = "https://avatars.githubusercontent.com/u/583231?v=4"
BOT_AVATAR = "https://img.icons8.com/emoji/48/robot-emoji.png"

def chat_bubble(inhalt, align="left", bgcolor="#F1F0F0", avatar_url=None):
    if inhalt is None:
        return
    align_css = "right" if align == "right" else "left"
    avatar_html = f"<img src='{avatar_url}' style='width: 30px; height: 30px; border-radius: 50%; margin-right: 10px;' />" if avatar_url else ""
    bubble_html = f"""
        <div style='text-align: {align_css}; margin: 10px 0; display: flex; flex-direction: {'row-reverse' if align=='right' else 'row'};'>
            {avatar_html}
            <div style='background-color: {bgcolor}; padding: 10px 15px; border-radius: 10px; max-width: 80%;'>
                {inhalt}
            </div>
        </div>
    """
    st.markdown(bubble_html, unsafe_allow_html=True)

def link_mit_chat_und_link(bot_text, url, key):
    st.session_state[key] = not st.session_state[key]
    if st.session_state[key]:
        link = f'<a href="{url}" target="_blank">🔍 Hier klicken, um zur Seite zu gelangen</a>'
        chat_bubble(link, align="left", bgcolor="#F1F0F0", avatar_url=BOT_AVATAR)

for nutzer, bot in st.session_state.chat_history:
    chat_bubble(nutzer, align="right", bgcolor="#DCF8C6", avatar_url=USER_AVATAR)
    chat_bubble(bot, align="left", bgcolor="#F1F0F0", avatar_url=BOT_AVATAR)


# Verarbeitung der Benutzereingabe
benutzereingabe = st.chat_input("Ihre Frage eingeben:")

    chat_bubble(benutzereingabe, align="right", bgcolor="#DCF8C6", avatar_url=USER_AVATAR)
    eingabe = benutzereingabe.strip().lower()

    if eingabe in ["hallo", "hi", "guten tag", "hey"]:
        willkommen = "Hallo und willkommen bei Wertgarantie! Was kann ich für Sie tun?"
        st.session_state.chat_history.append((benutzereingabe, willkommen))
        chat_bubble(willkommen, align="left", bgcolor="#F1F0F0", avatar_url=BOT_AVATAR)
    else:
        context = get_relevant_chunks(benutzereingabe)
        context_text = "\n".join([c[0] for c in context])

        # Multiround context from previous exchanges
        conversation_history = []
        for prev_user, prev_bot in st.session_state.chat_history[-6:]:  # letzte 6 Runden merken
            conversation_history.append({"role": "user", "content": prev_user})
            conversation_history.append({"role": "assistant", "content": prev_bot})

        messages = [
            {
                "role": "system",
                "content": (
                    "Sie sind ein professioneller Kundenservice-Chatbot für eine deutsche Versicherung. "
                    "Bitte antworten Sie ausschließlich auf Deutsch, in korrektem, höflichem Ton (durchgehend Siezen). "
                    "Achten Sie besonders auf Rechtschreibung, Grammatik und technische Fachbegriffe. "
                    "Ihre Antworten sollen klar, vertrauenswürdig und hilfreich sein. Geben Sie, falls relevant, zuerst eine hilfreiche Einschätzung und schlagen Sie danach passend ein Versicherungsprodukt vor."
                )
            }
        ] + conversation_history + [
            {
                "role": "user",
                "content": f"Relevante Informationen:\n{context_text}\n\nFrage: {user_input}"
            }
        ]

        response = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct:free",
            messages=messages
        )

        answer = response.choices[0].message.content

        st.session_state.chat_history.append((benutzereingabe, answer))
        st.chat_message("assistant").write(answer)
        chat_bubble(korrigiert, align="left", bgcolor="#F1F0F0", avatar_url=BOT_AVATAR)



st.markdown("""---
**Wählen Sie eine Kategorie:**
""")

col1 = st.columns(1)[0]
with col1:
    if st.button("Smartphone-, Waschmaschine-, Kamera-Versicherung", key="btn1"):
        st.session_state.show_sub_buttons = not st.session_state.show_sub_buttons

if st.session_state.show_sub_buttons:
    st.markdown("**Wählen Sie die Geräteversicherung aus:**")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("📱 Smartphone-Versicherung", key="sub1"):
            link_mit_chat_und_link("", "https://www.wertgarantie.de/versicherung#/smartphone", "show_link_smartphone")
        if st.button("💻 Notebook-Versicherung", key="sub2"):
            link_mit_chat_und_link("", "https://www.wertgarantie.de/versicherung#/notebook", "show_link_notebook")

    with col_b:
        if st.button("📷 Kamera-Versicherung", key="sub3"):
            link_mit_chat_und_link("", "https://www.wertgarantie.de/versicherung#/kamera", "show_link_kamera")
        if st.button("📺 Fernseher-Versicherung", key="sub4"):
            link_mit_chat_und_link("", "https://www.wertgarantie.de/versicherung#/fernseher", "show_link_tv")

col2, col3 = st.columns(2)
with col2:
    if st.button("Werkstätten", key="btn2"):
        link_mit_chat_und_link("", "https://www.wertgarantie.de/werkstattsuche", "show_link_werkstatt")
with col3:
    if st.button("Fachhändler", key="btn3"):
        link_mit_chat_und_link("", "https://www.wertgarantie.de/haendlersuche", "show_link_haendler")
