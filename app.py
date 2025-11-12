# app.py
"""
Analyse Prédictive + Chat IA (open-source) - Streamlit app
Fonctions :
- Upload CSV/XLSX
- Entraînement d'un modèle simple (LinearRegression)
- Dashboard Réel vs Prévu + export CSV
- Conseils automatiques (règles) + suggestions LLM (GPT4All si disponible)
- Chat avec historique sauvegardée par client
Notes :
- Si GPT4All n'est pas disponible sur le runtime, l'app utilise un fallback simple.
- Sauvegardes locales : dossiers 'uploaded_data' et 'conversations'
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.express as px
from datetime import datetime

# Try to import GPT4All; if unavailable, set flag and use fallback
try:
    from gpt4all import GPT4All
    GPT_AVAILABLE = True
except Exception:
    GPT_AVAILABLE = False

# ---- Configuration ----
st.set_page_config(page_title="Analyse IA Entreprise", layout="wide")
DATA_FOLDER = "uploaded_data"
CONV_FOLDER = "conversations"
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(CONV_FOLDER, exist_ok=True)

# ---- Helpers: paths & history ----
def conv_path(client: str) -> str:
    safe = client.strip().replace(" ", "_")
    return os.path.join(CONV_FOLDER, f"{safe}_conversations.json")

def load_history(client: str):
    p = conv_path(client)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history(client: str, history):
    p = conv_path(client)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def display_history(history, limit=50):
    # Affiche les derniers échanges
    for turn in history[-limit:]:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        ts = turn.get("ts", "")
        if role == "user":
            st.markdown(f"**Client** ({ts}) : {content}")
        else:
            st.markdown(f"**Assistant** ({ts}) : {content}")

# ---- LLM wrapper (open-source) ----
def generate_with_llm(prompt: str, max_tokens: int = 200) -> str:
    """
    Utilise GPT4All si disponible, sinon fallback heuristique.
    Remarque : le nom du modèle peut varier selon l'installation du runtime.
    """
    if GPT_AVAILABLE:
        try:
            # essaye un nom commun ; selon l'environnement, le nom peut être différent
            llm = GPT4All(model="gpt4all-lora-quantized")  
            out = llm.generate(prompt, max_tokens=max_tokens)
            # generate peut renvoyer une chaîne ou une structure ; on assure une string
            if isinstance(out, (list, tuple)):
                return "\n".join(str(x) for x in out)
            return str(out)
        except Exception as e:
            # si erreur, on bascule en fallback
            st.error("LLM error (open-source) : " + str(e))
            return fallback_ai_answer(prompt)
    else:
        return fallback_ai_answer(prompt)

def fallback_ai_answer(prompt: str) -> str:
    # Réponse de secours simple et utile
    lines = [
        "Réponse automatique (fallback) :",
        "- Vérifiez les produits dont les ventes baissent régulièrement.",
        "- Réduisez les stocks des produits faiblement demandés.",
        "- Augmentez le budget publicitaire sur les produits en croissance.",
        "- Analysez la saisonnalité et planifiez les promotions.",
        "- Pour une analyse plus fine, fournissez plus de données (périodes, promotions, canaux)."
    ]
    return "\n".join(lines)

# ---- UI header & sidebar ----
st.markdown("<h1 style='text-align:center;'>📊 Analyse Prédictive & Chat IA (Open-Source)</h1>", unsafe_allow_html=True)
st.write("Upload → Analyse → Dashboard → Conseils → Chat (historique sauvegardé).")

st.sidebar.header("Paramètres client / Branding")
client_name = st.sidebar.text_input("Nom de l'entreprise / client", value="client_exemple")
logo_url = st.sidebar.text_input("URL du logo (optionnel)", value="")
primary_color = st.sidebar.color_picker("Couleur principale", "#1f77b4")

# ---- Upload des données ----
st.header("1) Upload des données (CSV / Excel)")
uploaded_file = st.file_uploader("Choisis un fichier CSV ou Excel", type=['csv', 'xlsx', 'xls'])

if not uploaded_file:
    st.info("Upload un fichier CSV ou Excel pour commencer (ex : ventes, visites, dépenses...).")
    st.stop()

# Lire le fichier
try:
    if uploaded_file.name.lower().endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error("Erreur de lecture du fichier : " + str(e))
    st.stop()

st.subheader("Aperçu des données")
st.dataframe(df.head(20), use_container_width=True)

# Sauvegarde locale du dataset pour trace
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
safe_client = client_name.strip().replace(" ", "_") or "client_exemple"
saved_name = f"{DATA_FOLDER}/{safe_client}_{timestamp}_{uploaded_file.name}"
try:
    if uploaded_file.name.lower().endswith('.csv'):
        df.to_csv(saved_name, index=False)
    else:
        df.to_excel(saved_name, index=False)
except Exception:
    # si impossible d'écrire, on ignore proprement (ex : runtime sans persistance)
    pass

# ---- Analyse prédictive ----
st.header("2) Analyse Prédictive (régression simple)")
# repérer colonnes numériques
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) < 1:
    st.warning("Aucune colonne numérique détectée. L'analyse prédictive nécessite au moins une colonne numérique.")
    st.stop()

st.write("Colonnes numériques détectées :", numeric_cols)
target = st.selectbox("Choisir la colonne cible (y) à prédire", numeric_cols, index=max(0, len(numeric_cols)-1))
available_features = [c for c in numeric_cols if c != target]
features = st.multiselect("Choisir les features (X) — vide => toutes sauf la cible", available_features, default=available_features)

# Préparation des données
X = df[features].fillna(0)
y = df[target].fillna(0)

# Entraîner et prédire
if st.button("Lancer l'entraînement et prédiction"):
    with st.spinner("Entraînement du modèle (LinearRegression) ..."):
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            ml_model = LinearRegression()
            ml_model.fit(X_train, y_train)
            preds = ml_model.predict(X)
            df['Prediction'] = preds
            st.success("Modèle entraîné et prédictions calculées ✅")
        except Exception as e:
            st.error("Erreur durant l'entraînement : " + str(e))
            st.stop()

    # Dashboard : Réel vs Prévu
    st.subheader("3) Dashboard & Graphiques")
    try:
        fig = px.line(df.reset_index(), y=[target, 'Prediction'],
                      labels={'index':'Index','value':'Valeur'}, title='Réel vs Prévu')
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error("Erreur graphique : " + str(e))

    # Metrics simples
    if 'Prediction' in df.columns:
        diffs = df['Prediction'] - df[target]
        st.metric("Moyenne Erreur (Prediction - Réel)", f"{diffs.mean():.2f}")
        st.write("Aperçu résultats :")
        st.dataframe(df.head(30), use_container_width=True)

        # bouton d'export CSV
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        st.download_button("Télécharger résultats (CSV)", csv_bytes, file_name=f"{safe_client}_results_{timestamp}.csv", mime="text/csv")

    # ---- Conseils automatiques (règle + IA) ----
    st.header("4) Conseils Automatiques")
    if 'Prediction' in df.columns:
        mean_pred = float(np.mean(df['Prediction']))
        mean_real = float(np.mean(df[target]))
        if mean_pred > mean_real:
            st.info("Conseil (règle) : Prévision supérieure à la moyenne historique → vérifiez les stocks et préparez l'approvisionnement.")
        else:
            st.info("Conseil (règle) : Prévision inférieure à la moyenne historique → envisagez promotions / marketing pour stimuler les ventes.")
    else:
        st.info("Prédictions non disponibles → exécute l'entraînement pour obtenir des conseils.")

    st.write("Conseils par IA (open-source) :")
    # préparer prompt (résumé)
    try:
        small_preview = df.head(5).to_dict(orient='records')
        prompt_summary = f"Résumé des premières lignes pour le client {client_name}:\n{small_preview}\nDonne 5 conseils business concis et actionnables basés sur ces données."
    except Exception:
        prompt_summary = f"Donne 5 conseils business basés sur des données de ventes."

    ai_resp = generate_with_llm(prompt_summary, max_tokens=200)
    st.markdown(ai_resp)

    # ---- Chat IA avec historique ----
    st.header("5) Chat IA (historique & continuation)")
    st.write("Pose une question contextuelle. L'historique du client est sauvegardé et utilisé comme contexte.")

    # load history from file
    history = load_history(client_name)

    if st.checkbox("Afficher l'historique de conversation"):
        if history:
            display_history(history, limit=200)
        else:
            st.write("Aucune conversation trouvée pour ce client.")

    user_msg = st.text_area("Votre question", value="", height=120)
    if st.button("Envoyer la question"):
        if not user_msg.strip():
            st.warning("Écris une question avant d'envoyer.")
        else:
            # ajouter message utilisateur
            history.append({"role":"user", "content": user_msg, "ts": datetime.now().isoformat()})
            # préparer contexte (dernières lignes)
            context_lines = [f"{h['role']}: {h['content']}" for h in history[-20:]]
            context = "\n".join(context_lines)
            prompt = f"Contexte:\n{context}\n\nQuestion:\n{user_msg}\n\nRéponds de manière professionnelle, concise et actionnable (français)."

            answer = generate_with_llm(prompt, max_tokens=300)
            # afficher et sauvegarder
            st.markdown(f"**Assistant**: {answer}")
            history.append({"role":"assistant", "content": answer, "ts": datetime.now().isoformat()})
            save_history(client_name, history)

    st.markdown("---")
    st.write("Fin de l'analyse pour ce fichier.")
else:
    st.info("Upload un fichier CSV ou Excel pour commencer l'analyse.")
