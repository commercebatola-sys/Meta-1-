import streamlit as st
import openai
import os

# ---------------------------------------------------------
# Configuration API (clé à mettre dans Streamlit Secrets)
# ---------------------------------------------------------
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.set_page_config(page_title="Assistant IA Document", layout="wide")
st.title("📄 Assistant IA - Analyse & Questions")

# Section upload
uploaded_file = st.file_uploader("📁 Importer un document (PDF, TXT, DOCX)", type=["txt", "pdf", "docx"])

# Lire le document selon type
def read_file(file):
    if file.type == "text/plain":
        return file.read().decode("utf-8")

    elif file.type == "application/pdf":
        import PyPDF2
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    elif file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        import docx
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])

    return ""


document_text = ""

if uploaded_file:
    try:
        document_text = read_file(uploaded_file)
        st.success("Document importé avec succès !")
    except:
        st.error("Erreur lors de la lecture du document.")

# ---------------------------------------------------------
# Fonctions OpenAI
# ---------------------------------------------------------

def openai_answer(system, user, max_tokens=400):
    response = openai.ChatCompletion.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        max_tokens=max_tokens
    )
    return response.choices[0].message["content"]


# Résumé du document
def summarize_text(text, max_words=200):
    prompt = f"Résumé clair, structuré et simple, limité à {max_words} mots.\n\nTexte : {text}"

    return openai_answer(
        "Tu es un expert en résumé et analyse de texte.",
        prompt,
        max_tokens=350
    )

# Questions suggérées
def generate_questions(text):
    prompt = (
        "Génère 4 questions pertinentes que l'utilisateur pourrait poser "
        "après avoir lu ce document. Pas de réponses. Liste simple."
        f"\n\nDocument : {text}"
    )

    result = openai_answer(
        "Tu es un assistant spécialisé en analyse documentaire.",
        prompt,
        max_tokens=150
    )

    return result.split("\n")

# Réponse à une question
def answer_question(doc, question):
    prompt = (
        f"Voici le document :\n\n{doc}\n\n"
        f"Question de l'utilisateur : {question}\n\n"
        "IMPORTANT : Même si l'information n'est pas dans le document, "
        "tu dois répondre en utilisant tes propres connaissances professionnelles."
    )

    return openai_answer(
        "Tu es un expert polyvalent capable d'expliquer, analyser et conseiller.",
        prompt,
        max_tokens=500
    )


# ---------------------------------------------------------
# Interface : Résumé, Questions, Chat
# ---------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("📌 Résumé du document")
    if st.button("Générer le résumé"):
        if not document_text:
            st.warning("Importe d’abord un document.")
        else:
            summary = summarize_text(document_text)
            st.text_area("Résumé :", summary, height=250)

with col2:
    st.subheader("❓ Questions suggérées")
    if st.button("Générer des questions"):
        if not document_text:
            st.warning("Importe d’abord un document.")
        else:
            qs = generate_questions(document_text)
            for q in qs:
                st.write("- " + q)


st.subheader("💬 Poser une question")
user_question = st.text_input("Tape ta question ici…")

if st.button("Répondre"):
    if not user_question.strip():
        st.warning("Pose une question.")
    else:
        answer = answer_question(document_text, user_question)
        st.write("### 💡 Réponse :")
        st.write(answer)
