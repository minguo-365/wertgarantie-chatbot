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

def grammatik_korrigieren(text):
    try:
        response = requests.post(
            "https://api.languagetoolplus.com/v2/check",
            data={"text": text, "language": "de-DE"},
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        matches = response.json().get("matches", [])
        for match in reversed(matches):
            offset = match["offset"]
            length = match["length"]
            replacement = match["replacements"][0]["value"] if match["replacements"] else ""
            text = text[:offset] + replacement + text[offset+length:]
        return text
    except:
        return text

def entferne_nicht_deutsch(text):
    text = re.sub(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+', '', text)
    text = re.sub(r'Es tut mir leid, dazu habe ich leider keine Informationen\.', '', text)
    return text.strip()

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

if st.button("🩹 Verlauf löschen"):
    st.session_state.chat_history = []
    st.rerun()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def chat_bubble(inhalt, align="left", bgcolor="#F1F0F0", avatar_url=None):
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

USER_AVATAR = "https://avatars.githubusercontent.com/u/583231?v=4"
BOT_AVATAR = "https://img.icons8.com/emoji/48/robot-emoji.png"

def link_mit_chat_und_link(nutzer_text, bot_text, url):
    link = f'<a href="{url}" target="_blank">👉 Hier klicken, um zur Seite zu gelangen</a>'
    gesamt_text = f"{bot_text}<br>{link}"
    st.session_state.chat_history.append((nutzer_text, gesamt_text))
    chat_bubble(nutzer_text, align="right", bgcolor="#DCF8C6", avatar_url=USER_AVATAR)
    chat_bubble(gesamt_text, align="left", bgcolor="#F1F0F0", avatar_url=BOT_AVATAR)

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

    elif any(stichwort in eingabe for stichwort in ["versicherung", "schaden"]):
        antwort = (
            "WERTGARANTIE bietet verschiedene Versicherungen an, darunter Schutz für Smartphones, Tablets, Laptops, E-Bikes/Fahrräder, Hörgeräte sowie Haushalts- und Unterhaltungselektronik. "
            "Unsere Produkte bieten umfassenden Schutz vor Reparaturkosten, Diebstahl und technischen Defekten. Möchten Sie zu einem bestimmten Gerät mehr erfahren?"
        )
        st.session_state.chat_history.append((benutzereingabe, antwort))
        chat_bubble(antwort, align="left", bgcolor="#F1F0F0", avatar_url=BOT_AVATAR)

    else:
        verlauf = []
        for frage, antwort in st.session_state.chat_history[-6:]:
            verlauf.append({"role": "user", "content": frage})
            verlauf.append({"role": "assistant", "content": antwort})

        nachrichten = [
            {"role": "system", "content": "Sie sind ein professioneller Kundenservice-Chatbot. Bitte antworten Sie hilfreich und korrekt auf Deutsch, möglichst prägnant und höflich."}
        ] + verlauf + [{"role": "user", "content": benutzereingabe}]

        antwort = frage_openrouter(nachrichten)
        antwort = entferne_nicht_deutsch(antwort)
        korrigiert = grammatik_korrigieren(antwort)
        st.session_state.chat_history.append((benutzereingabe, korrigiert))
        chat_bubble(korrigiert, align="left", bgcolor="#F1F0F0", avatar_url=BOT_AVATAR)

# Immer anzeigen, nicht nur bei Begrüßung
st.markdown("""---
**Wählen Sie eine Kategorie:**
""")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Smartphone-Versicherung", key="btn1"):
        link_mit_chat_und_link(
            "Ich interessiere mich für eine Smartphone-Versicherung.",
            "Hier finden Sie Informationen zur Smartphone-Versicherung.",
            "https://www.wertgarantie.de/versicherung#/"
        )
with col2:
    if st.button("Waschmaschine-Versicherung", key="btn2"):
        link_mit_chat_und_link(
            "Ich möchte meine Waschmaschine versichern.",
            "Hier finden Sie Informationen zur Waschmaschinen-Versicherung.",
            "https://www.wertgarantie.de/versicherung#/"
        )
with col3:
    if st.button("Kamera-Versicherung", key="btn3"):
        link_mit_chat_und_link(
            "Ich möchte meine Kamera versichern.",
            "Hier finden Sie Informationen zur Kamera-Versicherung.",
            "https://www.wertgarantie.de/versicherung#/"
        )
