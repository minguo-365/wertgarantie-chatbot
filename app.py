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
option = st.radio(
    "Bitte wählen Sie eine Kategorie:",
    (
        "Autoversicherung",
        "Auslandskrankenschutz",
        "Reiserücktrittsversicherung",
        "Familienmitgliedschaft",
        "Hilfe zur Mitgliedskarte",
        "Kontakt zum Kundenservice"
    ),
    horizontal=True
)

st.write(f"Sie haben gewählt: **{option}**")

# Hier könntest du je nach Auswahl unterschiedliche Chatbot-Antworten generieren
if option == "Autoversicherung":
    st.info("Hier findest du Infos zur Autoversicherung...")
elif option == "Kontakt zum Kundenservice":
    st.info("Du kannst unseren Kundenservice per E-Mail oder Hotline kontaktieren...")

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

    if user_input.strip().lower() in ["hallo", "hi", "guten tag", "hey"]:
        welcome_reply = "Hallo und willkommen bei Wertgarantie! Was kann ich für Sie tun?"
        st.session_state.chat_history.append((user_input, welcome_reply))
        chat_bubble(welcome_reply, align="left", bgcolor="#F1F0F0", avatar_url=BOT_AVATAR)
    elif any(keyword in user_input.lower() for keyword in ["versicherung", "schaden melden"]):
        versicherung_reply = (
            "WERTGARANTIE bietet verschiedene Versicherungen an, darunter Schutz für Smartphones, Tablets, Laptops, E-Bikes/Fahrräder, Hörgeräte sowie Haushalts- und Unterhaltungselektronik."
            " Unsere Produkte bieten umfassenden Schutz vor Reparaturkosten, Diebstahl und technischen Defekten. Möchten Sie zu einem bestimmten Gerät mehr erfahren?"
        )
        st.session_state.chat_history.append((user_input, versicherung_reply))
        chat_bubble(versicherung_reply, align="left", bgcolor="#F1F0F0", avatar_url=BOT_AVATAR)
    else:
        conversation_history = []
        for prev_user, prev_bot in st.session_state.chat_history[-6:]:
            conversation_history.append({"role": "user", "content": prev_user})
            conversation_history.append({"role": "assistant", "content": prev_bot})

        messages = [
            {
                "role": "system",
                "content": (
                    "Sie sind ein professioneller Kundenservice-Chatbot. "
                    "Bitte antworten Sie hilfreich und korrekt auf Deutsch, möglichst prägnant und höflich."
                )
            }
        ] + conversation_history + [
            {"role": "user", "content": user_input}
        ]

        answer = ask_openrouter(messages)
        answer = remove_non_german(answer)
        corrected = correct_grammar_with_languagetool(answer)

        st.session_state.chat_history.append((user_input, corrected))
        chat_bubble(corrected, align="left", bgcolor="#F1F0F0", avatar_url=BOT_AVATAR)
