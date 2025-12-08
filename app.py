
# app.py
import streamlit as st
import fitz  # PyMuPDF
from openai import OpenAI
import tempfile
import os

# ===============================
# Clé API directe
# ===============================
OPENAI_API_KEY = "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"

client = OpenAI(api_key=OPENAI_API_KEY)

# ===============================
# Configuration Streamlit
# ===============================
st.set_page_config(
    page_title="Analyse Automatique de Documents Financiers",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Analyse Automatique de Documents Financiers")
st.markdown("Transformez vos PDF financiers en résumé clair et chiffré.")

# ===============================
# Sidebar Configuration
# ===============================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Sélection du modèle
    model = st.selectbox(
        "Modèle OpenAI",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0
    )
    
    # Limite de longueur du texte
    max_length = st.slider(
        "Longueur maximale du texte (caractères)",
        min_value=50000,
        max_value=200000,
        value=120000,
        step=10000
    )
    
    st.markdown("---")
    st.markdown("**Instructions :**")
    st.markdown("1. Uploadez votre PDF financier")
    st.markdown("2. Obtenez un résumé structuré")
    st.markdown("3. Posez des questions, même hors PDF")

# ===============================
# Fonctions principales
# ===============================
def extract_pdf_text(pdf_file, max_length=120000):
    """Extrait le texte d'un PDF avec repères de pages"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_path = tmp_file.name
        
        pdf = fitz.open(tmp_path)
        text = ""
        for i, page in enumerate(pdf, start=1):
            page_text = page.get_text()
            text += f"\n\n=== [PAGE {i}] ===\n" + page_text.strip()
        
        text = "\n".join(line.strip() for line in text.splitlines())
        if len(text) > max_length:
            text = text[:max_length]
            st.warning(f"⚠️ Texte tronqué à {max_length} caractères")
        os.unlink(tmp_path)
        return text, len(text)
    except Exception as e:
        st.error(f"❌ Erreur lecture PDF: {e}")
        return None, 0

def generate_summary(text, model="gpt-4o-mini"):
    """Génère un résumé financier structuré"""
    instructions = (
        "Tu es analyste financier. On te fournit le texte d'un document financier.\n"
        "Produis une synthèse **précise et chiffrée** en Markdown :\n"
        "- Société / Période / Devise\n"
        "- Résumé exécutif (5–8 lignes)\n"
        "- Chiffres clés (tableau Markdown)\n"
        "- Analyse\n"
        "- Références internes\n"
        "N'invente aucun chiffre si absent."
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": text}
            ],
            max_tokens=2000,
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"❌ Erreur génération résumé: {e}")
        return None

def answer_question(text, question, model="gpt-4o"):
    """Répond à une question spécifique ou générale"""
    instructions = (
        "Tu es analyste financier et assistant business. "
        "Réponds à la question posée sur le document ou de manière générale "
        "si l'information n'est pas dans le PDF. "
        "Si impossible, écris 'non précisé'."
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": f"Question : {question}\n\nTexte PDF :\n{text}"}
            ],
            max_tokens=1000,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"❌ Erreur réponse question: {e}")
        return None

# ===============================
# Interface principale
# ===============================
def main():
    tab1, tab2 = st.tabs(["📄 Upload & Analyse", "❓ Questions"])
    
    with tab1:
        st.header("📄 Upload et Analyse du PDF")
        uploaded_file = st.file_uploader("Choisissez votre PDF financier", type=['pdf'])
        
        if uploaded_file is not None:
            file_details = {
                "Nom": uploaded_file.name,
                "Taille": f"{uploaded_file.size / 1024:.1f} KB",
                "Type": uploaded_file.type
            }
            st.json(file_details)
            
            if st.button("🚀 Analyser le document"):
                with st.spinner("📖 Extraction en cours..."):
                    text, text_length = extract_pdf_text(uploaded_file, max_length)
                
                if text:
                    st.success(f"✅ Texte extrait : {text_length} caractères")
                    with st.expander("👁️ Aperçu du texte extrait"):
                        st.text(text[:1000] + "..." if len(text) > 1000 else text)
                    
                    with st.spinner("🤖 Génération du résumé..."):
                        summary = generate_summary(text, model)
                    
                    if summary:
                        st.success("✅ Résumé généré !")
                        st.subheader("📊 Résumé Financier")
                        st.markdown(summary)
                        st.session_state['pdf_text'] = text
                        st.session_state['summary'] = summary
                        st.download_button(
                            "💾 Télécharger le résumé (Markdown)",
                            data=summary,
                            file_name=f"resume_{uploaded_file.name.replace('.pdf','')}.md",
                            mime="text/markdown"
                        )
    
    with tab2:
        st.header("❓ Questions sur le Document")
        if 'pdf_text' not in st.session_state:
            st.info("ℹ️ Analysez d'abord un PDF dans l'onglet 'Upload & Analyse'")
        else:
            st.success("✅ Document prêt pour les questions")
            question = st.text_input("Posez votre question (ex: augmenter CA de 10%)")
            if question:
                if st.button("🔍 Rechercher la réponse"):
                    with st.spinner("🤖 Recherche en cours..."):
                        answer = answer_question(st.session_state['pdf_text'], question, model)
                    if answer:
                        st.success("✅ Réponse trouvée !")
                        st.markdown("**Question :** " + question)
                        st.markdown("**Réponse :**")
                        st.markdown(answer)
            
            st.subheader("💡 Questions suggérées")
            suggested_questions = [
                "Quel est le chiffre d'affaires ?",
                "Quelle est la marge nette ?",
                "Quels sont les principaux risques identifiés ?",
                "Quelle est la dette nette ?",
                "Quel est le cash flow opérationnel ?",
                "Comment augmenter mon chiffre d'affaires de 10% ?"
            ]
            for i, q in enumerate(suggested_questions):
                if st.button(f"❓ {q}", key=f"suggested_{i}"):
                    with st.spinner("🤖 Recherche en cours..."):
                        answer = answer_question(st.session_state['pdf_text'], q, model)
                    if answer:
                        st.success("✅ Réponse trouvée !")
                        st.markdown("**Question :** " + q)
                        st.markdown("**Réponse :**")
                        st.markdown(answer)

# Footer
st.markdown("---")
st.markdown(
    "**Note :** Vérifiez toujours les chiffres affichés et leurs pages d'origine. "
    "En cas d'ambiguïté, utilisez 'non précisé' et confirmez dans le document source."
)

if __name__ == "__main__":
    main()
