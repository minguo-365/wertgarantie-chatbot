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

def get_relevante_abschnitte(anfrage, k=3):
    anfrage_vektor = model.encode([anfrage])
    D, I = index.search(np.array(anfrage_vektor), k)
    return [(chunks[i], i) for i in I[0]]


def frage_openrouter(nachrichten):
    try:
        response = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct",
            messages=nachrichten
        )
        return response.choices[0].message.content
    except Exception as e:
        if "code': 402" in str(e).lower() or "insuffizien" in str(e).lower():
            try:
                response = client.chat.completions.create(
                    model="mistralai/mistral-7b-instruct:free",
                    messages=nachrichten
                )
                return response.choices[0].message.content
            except Exception as e2:
                return f"❌ Auch das kostenlose Modell schlug fehl: {e2}"
        else:
            return f"❌ OpenRouter Fehler: {e}"

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

benutzereingabe = st.chat_input("Ihre Frage eingeben:")
if benutzereingabe:
    chat_bubble(benutzereingabe, align="right", bgcolor="#DCF8C6", avatar_url=USER_AVATAR)
    eingabe = benutzereingabe.strip().lower()

    if eingabe in ["hallo", "hi", "guten tag", "hey"]:
        willkommen = "Hallo und willkommen bei Wertgarantie! Was kann ich für Sie tun?"
        st.session_state.chat_history.append((benutzereingabe, willkommen))
        chat_bubble(willkommen, align="left", bgcolor="#F1F0F0", avatar_url=BOT_AVATAR)

    else:
        verlauf = []
        for frage, antwort in st.session_state.chat_history[-6:]:
            if frage: verlauf.append({"role": "user", "content": frage})
            verlauf.append({"role": "assistant", "content": antwort})

        nachrichten = [
            {"role": "system", "content": (
                    "Sie sind ein professioneller Kundenservice-Chatbot für eine deutsche Versicherung. "
                    "Bitte antworten Sie ausschließlich auf Deutsch, in korrektem, höflichem Ton (durchgehend Siezen). "
                    "Achten Sie besonders auf Rechtschreibung, Grammatik und technische Fachbegriffe. "
                    "Ihre Antworten sollen klar, vertrauenswürdig und hilfreich sein. Geben Sie, falls relevant, zuerst eine hilfreiche Einschätzung und schlagen Sie danach passend ein Versicherungsprodukt vor."
                )}] + verlauf + [{"role": "user", "content": benutzereingabe}]

        antwort = frage_openrouter(nachrichten)
        st.session_state.chat_history.append((benutzereingabe, antwort))
        chat_bubble(antwort, align="left", bgcolor="#F1F0F0", avatar_url=BOT_AVATAR)


if not st.session_state.get('chat_history', []):
    st.markdown("""---
**Wählen Sie eine Kategorie:**
""")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Versicherung", key="btn1"):
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
with col2:
    if st.button("Werkstätten", key="btn2"):
        link_mit_chat_und_link("", "https://www.wertgarantie.de/werkstattsuche", "show_link_werkstatt")
with col3:
    if st.button("Fachhändler", key="btn3"):
        link_mit_chat_und_link("", "https://www.wertgarantie.de/haendlersuche", "show_link_haendler")

#col4 = st.columns(1)[0]
#col4, col5 = st.columns(2)
#with col4:
    #if st.button("FAQ", key="btn1"):
        #link_mit_chat_und_link("", "https://www.wertgarantie.de/service/haeufige-fragen?question=116241&title=was-passiert-wenn-ein-schaden-eintritt", "show_link_FAQ")
#with col5:
    #if st.button("Handy Erste Hilfe", key="btn2"):
        #link_mit_chat_und_link("", "https://www.wertgarantie.de/ratgeber/elektronik/smartphone/selbst-reparieren", "show_link_handy_erste_hilfe")
