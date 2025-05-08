import streamlit as st
import pandas as pd
import os
import time
import faiss
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from openai import OpenAI
from sentence_transformers import SentenceTransformer

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
        # Direkter Aufruf des kostenlosen Modells
        response = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct:free",  # Kostenlose Version
            messages=nachrichten
        )
        return response.choices[0].message.content  # Gibt die generierte Antwort zurück
    
    except Exception as e:
        return f"✘ Anfrage an kostenloses Modell fehlgeschlagen: {e}"  

@st.cache_data
def train_glm_model():
    df = pd.DataFrame({
        'Alter': [25, 45, 30, 60, 35, 22, 50],
        'Geraetewert': [800, 500, 1200, 400, 1000, 950, 350],
        'Marke': ['Apple', 'Samsung', 'Apple', 'Andere', 'Apple', 'Samsung', 'Andere'],
        'Schadenhistorie': [0, 1, 0, 1, 0, 1, 0],
        'Schadenhoehe': [0, 150, 0, 300, 0, 100, 0]
    })
    df = pd.get_dummies(df, columns=['Marke'], drop_first=True)
    formula = 'Schadenhoehe ~ Alter + Geraetewert + Schadenhistorie + Marke_Apple + Marke_Samsung'
    tweedie = sm.families.Tweedie(var_power=1.5, link=sm.families.links.log())
    glm_model = smf.glm(formula=formula, data=df, family=tweedie).fit()
    return glm_model

glm_model = train_glm_model()


# Layout mit zwei Spalten
col1, col2 = st.columns([1, 6])  # Erste Spalte schmal für Bild, zweite breit für Text

with col1:
    st.image("Bild.jpeg", width=360)  # Passe ggf. den Pfad oder Dateinamen an

with col2:
    st.markdown("# Wertgarantie Chatbot")  # kleinere, aber schöne Schriftgröße
#st.image("Bild.jpeg", width=260)
#st.markdown("<h4>Wertgarantie Chatbot</h4>", unsafe_allow_html=True)


if st.button("Verlauf löschen"):
    st.session_state.clear()
    st.rerun()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "frage_schritt" not in st.session_state:
    st.session_state.frage_schritt = 0

link_keys = [
    "show_link_smartphone", "show_link_notebook", "show_link_kamera", "show_link_tv",
    "show_link_werkstatt", "show_link_haendler", "show_link_ersteHilfe", "show_link_haushaltSelbstreparatur"
]
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

user_input = st.chat_input("Ihre Frage eingeben:")
if user_input:
    eingabe = user_input.strip().lower()

    if eingabe == "handyversicherung":
        st.session_state.frage_schritt += 1

    elif eingabe in ["hallo", "hi", "guten tag", "hey"]:
        willkommen = "Hallo und willkommen bei Wertgarantie! Was kann ich für Sie tun?"
        st.session_state.chat_history.append((user_input, willkommen))
        chat_bubble(user_input, align="right", bgcolor="#DCF8C6", avatar_url=USER_AVATAR)
        chat_bubble(willkommen, align="left", bgcolor="#F1F0F0", avatar_url=BOT_AVATAR)

    else:
        verlauf = []
        for frage, antwort in st.session_state.chat_history[-6:]:
            if frage:
                verlauf.append({"role": "user", "content": frage})
            verlauf.append({"role": "assistant", "content": antwort})

        context = get_relevante_abschnitte(user_input)
        context_text = "\n".join([c[0] for c in context])

        nachrichten = [
            {"role": "system", "content": (
                "Du bist ein kompetenter deutscher Kundenservice-Chatbot für ein Versicherungsunternehmen. "
                "Antworten bitte stets auf Deutsch, höflich und verständlich. Halte dich an technische und rechtliche Fakten, "
                "aber sprich den Nutzer ruhig menschlich und freundlich an."
            )}
        ] + verlauf + [
            {"role": "user", "content": f"Relevante Inhalte:\n{context_text}\n\nFrage: {user_input}"}
        ]

        chat_bubble(user_input, align="right", bgcolor="#DCF8C6", avatar_url=USER_AVATAR)

        antwort_placeholder = st.empty()
        antwort_placeholder.markdown(f"""
        <div style="display: flex; align-items: center; margin: 10px 0;">
          <img src="{BOT_AVATAR}" style="width: 30px; height: 30px; border-radius: 50%; margin-right: 10px;" />
          <div class="dot-flashing"></div>
        </div>

        <style>
        .dot-flashing {{
          position: relative;
          width: 10px;
          height: 10px;
          border-radius: 5px;
          background-color: #999;
          color: #999;
          animation: dot-flashing 1s infinite linear alternate;
        }}

        .dot-flashing::before, .dot-flashing::after {{
          content: "";
          display: inline-block;
          position: absolute;
          top: 0;
          width: 10px;
          height: 10px;
          border-radius: 5px;
          background-color: #999;
          color: #999;
        }}

        .dot-flashing::before {{
          left: -15px;
          animation: dot-flashing 1s infinite alternate;
        }}

        .dot-flashing::after {{
          left: 15px;
          animation: dot-flashing 1s infinite alternate;
          animation-delay: 0.5s;
        }}

        @keyframes dot-flashing {{
          0% {{ background-color: #ccc; }}
          50% {{ background-color: #999; }}
          100% {{ background-color: #ccc; }}
        }}
        </style>
        """, unsafe_allow_html=True)

        
        antwort = frage_openrouter(nachrichten)
        antwort_placeholder.empty()

        chat_bubble(antwort, align="left", bgcolor="#F1F0F0", avatar_url=BOT_AVATAR)
        st.session_state.chat_history.append((user_input, antwort))


if st.session_state.frage_schritt > 0:
    st.subheader("📋 Bitte beantworten Sie folgende Fragen:")

    with st.form(key="formular1"):
        if st.session_state.frage_schritt == 1:
            alter = st.text_input("1️⃣ Wie alt sind Sie?", key="alter_input")
            submitted = st.form_submit_button("Weiter ➔")
            if submitted and alter.isdigit() and 16 <= int(alter) <= 100:
                st.session_state.alter = int(alter)
                st.session_state.frage_schritt = 2
                st.rerun()
            elif submitted:
                st.warning("Bitte geben Sie ein Alter zwischen 16 und 100 ein.")

        elif st.session_state.frage_schritt == 2:
            wert = st.text_input("2️⃣ Wie viel kostet Ihr Handy? (€)", key="wert_input")
            submitted = st.form_submit_button("Weiter ➔")
            if submitted and wert.isdigit() and 50 <= int(wert) <= 2000:
                st.session_state.geraetewert = int(wert)
                st.session_state.frage_schritt = 3
                st.rerun()
            elif submitted:
                st.warning("Bitte geben Sie einen Wert zwischen 50 und 2000 ein.")

        elif st.session_state.frage_schritt == 3:
            marke = st.text_input("3️⃣ Welche Marke ist Ihr Handy? (Apple, Samsung, Andere)", key="marke_input")
            submitted = st.form_submit_button("Weiter ➔")
            if submitted and marke.capitalize() in ["Apple", "Samsung", "Andere"]:
                st.session_state.marke = marke.capitalize()
                st.session_state.frage_schritt = 4
                st.rerun()
            elif submitted:
                st.warning("Bitte geben Sie Apple, Samsung oder Andere ein.")

        elif st.session_state.frage_schritt == 4:
            schaden = st.text_input("4️⃣ Gab es im letzten Jahr einen Schaden? (Ja/Nein)", key="schaden_input")
            submitted = st.form_submit_button("📊 Tarif berechnen")
            if submitted and schaden.capitalize() in ["Ja", "Nein"]:
                st.session_state.schadenhistorie = schaden.capitalize()
                st.session_state.frage_schritt = 5
                st.rerun()
            elif submitted:
                st.warning("Bitte antworten Sie mit Ja oder Nein.")

    if st.session_state.frage_schritt == 5:
        daten = pd.DataFrame([{
            'Alter': st.session_state.alter,
            'Geraetewert': st.session_state.geraetewert,
            'Schadenhistorie': 1 if st.session_state.schadenhistorie == 'Ja' else 0,
            'Marke_Apple': 1 if st.session_state.marke == 'Apple' else 0,
            'Marke_Samsung': 1 if st.session_state.marke == 'Samsung' else 0
        }])

        erwartete_schadenhoehe = glm_model.predict(daten)[0]
        tarif_monatlich = (erwartete_schadenhoehe * 1.3) / 12

        st.success(f"✅ Ihre geschätzte monatliche Prämie beträgt: **{tarif_monatlich:.2f} €**")

        if st.button("🔄 Neue Berechnung starten"):
            for key in [
                "frage_schritt", "alter", "geraetewert", "marke", "schadenhistorie", "chat_history",
                "show_link_smartphone", "show_link_notebook", "show_link_kamera", "show_link_tv",
                "show_link_werkstatt", "show_link_haendler", "show_link_ersteHilfe", "show_link_haushaltSelbstreparatur"
            ]:
                st.session_state.pop(key, None)
            st.rerun()

if st.session_state.get("frage_schritt", 0) == 0 and not st.session_state.get("chat_history", []):
    st.markdown("""
    ---
    **Wählen Sie eine Kategorie:**
    """)

    show_versicherung = st.toggle("Versicherung", key="toggle_versicherung")
    show_werkstaetten = st.toggle("Werkstätten", key="toggle_werkstaetten")
    show_haendler = st.toggle("Fachhändler", key="toggle_haendler")
    show_erstehilfe = st.toggle("Erste Hilfe", key="toggle_erstehilfe")

    if show_versicherung:
        st.markdown("**Versicherung auswählen:**")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Smartphone-Versicherung", key="sub1"):
                link_mit_chat_und_link("", "https://www.wertgarantie.de/versicherung/smartphone#/buchung/1", "show_link_smartphone")
            if st.button("Notebook-Versicherung", key="sub2"):
                link_mit_chat_und_link("", "https://www.wertgarantie.de/versicherung#/buchung/1", "show_link_notebook")
        with col_b:
            if st.button("Kamera-Versicherung", key="sub3"):
                link_mit_chat_und_link("", "https://www.wertgarantie.de/versicherung#/buchung/1", "show_link_kamera")
            if st.button("E-Bike/Fahrrad", key="sub4"):
                link_mit_chat_und_link("", "https://www.linexo.de/versicherung", "show_link_tv")

    if show_werkstaetten:
        link_mit_chat_und_link("", "https://www.wertgarantie.de/werkstattsuche", "show_link_werkstatt")

    if show_haendler:
        link_mit_chat_und_link("", "https://www.wertgarantie.de/haendlersuche", "show_link_haendler")

    if show_erstehilfe:
        st.markdown("**Erste Hilfe auswählen:**")
        col_c, col_d = st.columns(2)
        with col_c:
            if st.button("Handy Selbstreparatur", key="sub5"):
                link_mit_chat_und_link("", "https://www.wertgarantie.de/ratgeber/elektronik/smartphone/selbst-reparieren", "show_link_ersteHilfe")
            if st.button("Haushalt Selbstreparatur", key="sub6"):
                link_mit_chat_und_link("", "https://www.wertgarantie.de/ratgeber/elektronik/haushalt-garten/selbst-reparieren", "show_link_haushaltSelbstreparatur")
                    
#col4 = st.columns(1)[0]
#col4, col5 = st.columns(2)
#with col4:
    #if st.button("FAQ", key="btn1"):
        #link_mit_chat_und_link("", "https://www.wertgarantie.de/service/haeufige-fragen?question=116241&title=was-passiert-wenn-ein-schaden-eintritt", "show_link_FAQ")
#with col5:
    #if st.button("Handy Erste Hilfe", key="btn2"):
        #link_mit_chat_und_link("", "https://www.wertgarantie.de/ratgeber/elektronik/smartphone/selbst-reparieren", "show_link_handy_erste_hilfe")

