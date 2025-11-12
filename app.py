# app.py
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.express as px
from datetime import datetime

# ---- Config ----
st.set_page_config(page_title="Analyse IA Entreprise", layout="wide")
DATA_FOLDER = "uploaded_data"
CONV_FOLDER = "conversations"
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(CONV_FOLDER, exist_ok=True)

st.markdown("<h1 style='text-align:center;'>📊 Analyse Prédictive & Chat IA (Open-Source)</h1>", unsafe_allow_html=True)
st.write("Interface: upload -> analyse -> dashboard -> conseils -> chat avec historique.")

# ---- Sidebar: client identification + branding ----
st.sidebar.header("Paramètres client")
client_name = st.sidebar.text_input("Nom de l'entreprise / client", value="client_exemple")
logo_url = st.sidebar.text_input("URL du logo (optionnel)")
primary_color = st.sidebar.color_picker("Couleur principale", "#1f77b4")

# function: path for client conversation file
def conv_path(client):
    safe = client.replace(" ", "_")
    return os.path.join(CONV_FOLDER, f"{safe}_conversations.json")

# Load history
def load_history(client):
    p = conv_path(client)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(client, history):
    p = conv_path(client)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# ---- File upload and data preview ----
st.header("1. Upload des données (CSV / Excel)")
uploaded_file = st.file_uploader("Choisis un fichier CSV ou Excel", type=['csv', 'xlsx', 'xls'])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error("Erreur lecture fichier: " + str(e))
        st.stop()

    st.subheader("Aperçu des données")
    st.dataframe(df.head(20), use_container_width=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_client = client_name.replace(" ", "_")
    fname = f"{DATA_FOLDER}/{safe_client}_{timestamp}_{uploaded_file.name}"
    if uploaded_file.name.endswith('.csv'):
        df.to_csv(fname, index=False)
    else:
        df.to_excel(fname, index=False)

    # ---- Analyse prédictive ----
    st.header("2. Analyse Prédictive (régression linéaire)")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) < 1:
        st.warning("Aucune colonne numérique trouvée. L'analyse prédictive nécessite au moins une colonne numérique.")
    else:
        st.write("Colonnes numériques détectées :", numeric_cols)
        target = st.selectbox("Choisir la colonne cible à prédire (y)", numeric_cols, index=len(numeric_cols)-1)
        features = st.multiselect("Choisir colonnes d'entrée (X) — si vide => toutes sauf cible", [c for c in numeric_cols if c != target])
        if len(features) == 0:
            features = [c for c in numeric_cols if c != target]
        st.write("Colonnes utilisées comme features :", features)

        X = df[features].fillna(0)
        y = df[target].fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X)
        df['Prediction'] = preds

        st.success("Modèle entraîné — prédictions ajoutées au tableau.")

        # ---- Dashboard & Graphiques ----
        st.subheader("3. Dashboard & Graphiques")
        fig = px.line(df.reset_index(), y=[target, 'Prediction'], labels={'index':'Index', 'value':'Valeur'}, title='Réel vs Prévu')
        st.plotly_chart(fig, use_container_width=True)

        diffs = df['Prediction'] - df[target]
        st.metric("Moyenne Erreur (Prediction - Réel)", f"{diffs.mean():.2f}")
        st.dataframe(df.head(30), use_container_width=True)

        st.download_button("Télécharger résultats (CSV)", df.to_csv(index=False), file_name=f"{safe_client}_results_{timestamp}.csv", mime="text/csv")

    # ---- Conseils automatiques ----
    st.header("4. Conseils Automatiques")
    mean_pred = float(np.mean(preds)) if 'preds' in locals() else None
    if mean_pred is not None:
        if mean_pred > y.mean():
            st.info("Conseil (règle): Les prévisions sont supérieures à la moyenne historique -> vérifier stocks & préparer approvisionnement.")
        else:
            st.info("Conseil (règle): Les prévisions sont basses -> envisager promotions/marketing pour stimuler les ventes.")

    st.write("Conseils par IA (fallback heuristique) :")
    st.write("- Vérifier produits avec baisse de ventes.\n- Réduire stock pour produits à faible demande.\n- Augmenter pub pour produits en hausse.\n- Vérifier saisonnalité.\n- Préparer plan de promotion pour prochaines périodes.")

    # ---- Chat IA avec historique ----
    st.header("5. Chat IA (historique & continuation)")
    history = load_history(client_name)
    if st.checkbox("Afficher l'historique de conversation"):
        if history:
            for turn in history[-50:]:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role == "user":
                    st.markdown(f"**Client**: {content}")
                else:
                    st.markdown(f"**Assistant**: {content}")
        else:
            st.write("Aucune conversation trouvée.")

    user_msg = st.text_area("Votre question", value="", height=100)
    if st.button("Envoyer la question"):
        if not user_msg.strip():
            st.warning("Écris une question.")
        else:
            history.append({"role":"user", "content": user_msg, "ts": datetime.now().isoformat()})
            # Fallback automatique
            answer = "Réponse automatique : analyse les données et vérifie les produits à marge basse ou forte demande."
            st.markdown(f"**Assistant**: {answer}")
            history.append({"role":"assistant", "content": answer, "ts": datetime.now().isoformat()})
            save_history(client_name, history)

    st.markdown("---")
    st.write("Fin de l'analyse pour ce fichier.")

else:
    st.info("Upload un fichier CSV ou Excel pour commencer l'analyse.")
