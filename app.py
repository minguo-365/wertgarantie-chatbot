# streamlit_app.py
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

def get_relevant_chunks(query, k=3):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), k)
    return [(chunks[i], i) for i in I[0]]

def correct_grammar_with_languagetool(text):
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

def remove_non_german(text):
    text = re.sub(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+', '', text)
    text = re.sub(r'Es tut mir leid, dazu habe ich leider keine Informationen\.', '', text)
    return text.strip()

def ask_openrouter(messages):
    try:
        response = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct",
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        if "code': 402" in str(e).lower() or "insuffizien" in str(e).lower():
            try:
                response = client.chat.completions.create(
                    model="mistralai/mistral-7b-instruct:free",
                    messages=messages
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

def chat_bubble(content, align="left", bgcolor="#F1F0F0", avatar_url=None):
    align_css = "right" if align == "right" else "left"
    avatar_html = f"<img src='{avatar_url}' style='width: 30px; height: 30px; border-radius: 50%; margin-right: 10px;' />" if avatar_url else ""
    bubble_html = f"""
        <div style='text-align: {align_css}; margin: 10px 0; display: flex; flex-direction: {'row-reverse' if align=='right' else 'row'};'>
            {avatar_html}
            <div style='background-color: {bgcolor}; padding: 10px 15px; border-radius: 10px; max-width: 80%;'>
                {content}
            </div>
        </div>
    """
    st.markdown(bubble_html, unsafe_allow_html=True)

USER_AVATAR = "https://avatars.githubusercontent.com/u/583231?v=4"
BOT_AVATAR = "https://img.icons8.com/emoji/48/robot-emoji.png"

for user_msg, bot_msg in st.session_state.chat_history:
    chat_bubble(user_msg, align="right", bgcolor="#DCF8C6", avatar_url=USER_AVATAR)
    chat_bubble(bot_msg, align="left", bgcolor="#F1F0F0", avatar_url=BOT_AVATAR)

user_input = st.chat_input("Ihre Frage eingeben:")
if user_input:
    chat_bubble(user_input, align="right", bgcolor="#DCF8C6", avatar_url=USER_AVATAR)

    benutzereingabe = user_input.strip().lower()

    # Begrüßung erkennen
    if benutzereingabe in ["hallo", "hi", "guten tag", "hey"]:
        welcome_reply = "Hallo und willkommen bei Wertgarantie! Was kann ich für Sie tun?"
        st.session_state.chat_history.append((user_input, welcome_reply))
        chat_bubble(welcome_reply, align="left", bgcolor="#F1F0F0", avatar_url=BOT_AVATAR)

# 👉 Buttons NUR bei Begrüßung anzeigen（Eintrag in den Chat + Weiterleitung）
def link_mit_chat_und_sprung(nachricht_user, nachricht_bot, url):
    st.session_state.chat_history.append((nachricht_user, nachricht_bot))
    chat_bubble(nachricht_user, align="right", bgcolor="#DCF8C6", avatar_url=USER_AVATAR)
    chat_bubble(nachricht_bot, align="left", bgcolor="#F1F0F0", avatar_url=BOT_AVATAR)
    # Automatische Weiterleitung (neuer Tab)
    st.markdown(f"""
        <meta http-equiv="refresh" content="0;url={url}" />
        <script>
            window.open("{url}", "_blank");
        </script>
    """, unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Smartphone-, Tablet-, Notebook-Versicherung", key="btn1"):
        link_mit_chat_und_sprung(
            "Ich interessiere mich für eine Smartphone-Versicherung.",
            "Hier finden Sie Informationen zur Smartphone-, Tablet- oder Notebook-Versicherung.",
            "https://www.wertgarantie.de/versicherung#/"
        )

with col2:
    if st.button("Waschmaschine-, Kaffeevollautomat-Versicherung", key="btn2"):
        link_mit_chat_und_sprung(
            "Ich möchte meine Waschmaschine oder meinen Kaffeevollautomaten versichern.",
            "Hier finden Sie Informationen zur Versicherung Ihrer Haushaltsgeräte.",
            "https://www.wertgarantie.de/versicherung#/"
        )

with col3:
    if st.button("Smartwatch-, Hörgerät-, Kamera-Versicherung", key="btn3"):
        link_mit_chat_und_sprung(
            "Ich benötige eine Versicherung für meine Smartwatch, Kamera oder mein Hörgerät.",
            "Hier finden Sie Schutzangebote für Smartwatches, Kameras und mehr.",
            "https://www.wertgarantie.de/versicherung#/"
        )

col4, col5, col6 = st.columns(3)
with col4:
    if st.button("Schaden melden", key="btn4"):
        link_mit_chat_und_sprung(
            "Ich möchte einen Schaden melden.",
            "Kein Problem – wir leiten Sie direkt zum Schadenformular weiter.",
            "https://www.wertgarantie.de/service/schaden-melden"
        )

with col5:
    if st.button("FAQ", key="btn5"):
        link_mit_chat_und_sprung(
            "Wo finde ich häufig gestellte Fragen (FAQ)?",
            "Hier finden Sie Antworten auf häufig gestellte Fragen.",
            "https://www.wertgarantie.de/service/haeufige-fragen"
        )

with col6:
    if st.button("Kontakt", key="btn6"):
        link_mit_chat_und_sprung(
            "Ich möchte den Kundenservice kontaktieren.",
            "Hier finden Sie unsere Kontaktmöglichkeiten.",
            "https://www.wertgarantie.de/service/kontakt"
        )

    # Versicherung oder Schadenmeldung erkennen
    elif any(stichwort in benutzereingabe for stichwort in ["versicherung", "schaden melden"]):
        antwort = (
            "WERTGARANTIE bietet verschiedene Versicherungen an, darunter Schutz für Smartphones, Tablets, Laptops, E-Bikes/Fahrräder, Hörgeräte sowie Haushalts- und Unterhaltungselektronik. "
            "Unsere Produkte bieten umfassenden Schutz vor Reparaturkosten, Diebstahl und technischen Defekten. Möchten Sie zu einem bestimmten Gerät mehr erfahren?"
        )
        st.session_state.chat_history.append((user_input, antwort))
        chat_bubble(antwort, align="left", bgcolor="#F1F0F0", avatar_url=BOT_AVATAR)

    # Standardantwort über OpenRouter (GPT)
    else:
        kontextverlauf = []
        for frage, antwort in st.session_state.chat_history[-6:]:
            kontextverlauf.append({"role": "user", "content": frage})
            kontextverlauf.append({"role": "assistant", "content": antwort})

        nachrichten = [
            {
                "role": "system",
                "content": (
                    "Sie sind ein professioneller Kundenservice-Chatbot. "
                    "Bitte antworten Sie hilfreich und korrekt auf Deutsch, möglichst prägnant und höflich."
                )
            }
        ] + kontextverlauf + [{"role": "user", "content": user_input}]

        antwort = ask_openrouter(nachrichten)
        antwort = remove_non_german(antwort)
        korrigiert = correct_grammar_with_languagetool(antwort)
        st.session_state.chat_history.append((user_input, korrigiert))
        chat_bubble(korrigiert, align="left", bgcolor="#F1F0F0", avatar_url=BOT_AVATAR)

