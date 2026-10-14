#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
================================================================================
    🏠 AGENT IMMOBILIER IA - V13
================================================================================
Ce script est une application Streamlit complète pour l'estimation immobilière.
Il utilise une architecture de Machine Learning hybride (RandomForest + XGBoost)
pour prédire les prix avec une grande précision.

ARCHITECTURE DU CODE :
1. Imports & Configuration : Initialisation de l'environnement.
2. Fonctions Core : Chargement, nettoyage et filtrage des données.
3. Moteur IA : Pipeline d'entraînement, optimisation et validation.
4. Interface Utilisateur : Gestion des onglets Estimation et Entraînement.

AUTEUR : [Neil LEGENDRE-FERREIRA DA COSTA (back-end)/ Rachelle ETOO AZOMBO (front-end)]
DATE : 2025
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import plotly.express as px
import plotly.graph_objects as go
import warnings
import io
import os

# --- Imports Scikit-learn & XGBoost ---
# Ces bibliothèques sont le coeur du moteur d'intelligence artificielle.
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor

# --- Configuration de la page Streamlit ---
st.set_page_config(page_title="IA Immobilier V13", page_icon="🏢", layout="wide")
warnings.filterwarnings('ignore')  # On ignore les avertissements mineurs pour ne pas polluer l'interface

# --- CSS Personnalisé ---
# Amélioration visuelle de l'interface (boutons, cadres de résultats)
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { background-color: #2c3e50; color: white; border-radius: 5px; width: 100%; font-weight: bold; }
    .metric-box { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 5px solid #e67e22; text-align: center; }
    .info-box { background-color: #e8f6f3; padding: 15px; border-radius: 8px; border: 1px solid #1abc9c; color: #0e6251; }
    </style>
    """, unsafe_allow_html=True)

# Nom du fichier pour la persistance du modèle
MODELE_FILE = 'modele_immo_v13.joblib'


# ============================================================================
# 1. FONCTIONS CORE (Chargement & Nettoyage)
# ============================================================================

@st.cache_data
def charger_donnees_expert(fichier_csv):
    """
    Charge et pré-nettoie le jeu de données CSV.

    POURQUOI CETTE FONCTION ?
    Les fichiers CSV bruts contiennent souvent des erreurs (séparateurs, doublons, espaces).
    Cette fonction agit comme un filtre pour garantir que l'IA reçoit des données propres.

    ACTIONS :
    1. Détection automatique du séparateur (virgule ou point-virgule).
    2. Suppression des doublons stricts pour éviter la fuite de données (Data Leakage).
    3. Standardisation des textes (Majuscules/Minuscules) pour éviter que 'Paris' != 'paris'.
    """
    try:
        # Tentative de lecture avec ',' puis ';' si échec
        try:
            df = pd.read_csv(fichier_csv, sep=',')
            if len(df.columns) < 5: raise ValueError
        except:
            fichier_csv.seek(0)
            df = pd.read_csv(fichier_csv, sep=';')

        # Dédoublonnage : Crucial pour ne pas avoir un score R² artificiel de 100%
        nb_init = len(df)
        df = df.drop_duplicates()
        nb_dedup = len(df)
        delta_dedup = nb_init - nb_dedup

        # Nettoyage des chaînes de caractères (strip espaces, format Titre)
        cols_txt = ['country', 'city', 'property_type']
        for c in cols_txt:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip().str.title()

        # Préparation des dictionnaires pour les listes déroulantes de l'interface
        villes_map = {}
        if 'country' in df.columns and 'city' in df.columns:
            for pays in sorted(df['country'].unique()):
                villes = sorted(df[df['country'] == pays]['city'].unique())
                villes_map[pays] = villes
        types = sorted(df['property_type'].unique()) if 'property_type' in df.columns else []

        return villes_map, types, df, nb_init, delta_dedup

    except Exception as e:
        st.error(f"Erreur Critique lors du chargement : {e}")
        return {}, [], None, 0, 0


def filtrer_outliers(df):
    """
    Supprime les valeurs aberrantes (Outliers) du dataset.

    POURQUOI ?
    Des biens à 1€ ou à 1 milliard d'euros (erreurs de saisie ou châteaux exceptionnels)
    peuvent fausser complètement l'apprentissage du modèle. On les retire pour
    que l'IA se concentre sur le "vrai" marché.

    MÉTHODE :
    On garde uniquement les prix compris entre le 1er centile (1%) et le 99ème centile (99%).
    """
    if 'price' not in df.columns: return df, 0, None, None
    q_low = df['price'].quantile(0.01)
    q_high = df['price'].quantile(0.99)
    df_filtered = df[(df['price'] > q_low) & (df['price'] < q_high)]
    return df_filtered, len(df) - len(df_filtered), q_low, q_high


# ============================================================================
# 2. MOTEUR D'INTELLIGENCE ARTIFICIELLE (Entraînement)
# ============================================================================

def train_hybrid_model(df, progress_bar, rf_trees, xgb_n_est, xgb_lr, xgb_depth, cv_folds):
    """
    Orchestre l'entraînement complet du modèle Hybride.

    POURQUOI UN MODÈLE HYBRIDE ?
    - RandomForest est excellent pour la stabilité et évite le sur-apprentissage.
    - XGBoost est imbattable pour la précision sur les détails fins.
    Combiner les deux (VotingRegressor) permet d'avoir le meilleur des deux mondes.

    PROCESSUS TECHNIQUE :
    1. Feature Engineering : Création de la variable 'Age' du bien.
    2. Log-Transformation : On prédit le log(Prix) pour réduire l'impact des très gros prix.
    3. Pipeline : Enchaînement du pré-traitement (OneHotEncoder) et du modèle.
    4. GridSearchCV : Test de toutes les combinaisons d'hyperparamètres pour trouver la meilleure.
    """

    # 1. Préparation des données (Feature Engineering)
    df_proc = df.copy()
    req_cols = ['property_size_sqft', 'constructed_year', 'rooms', 'bathrooms', 'garage', 'garden',
                'neighbourhood_rating', 'connectivity_score', 'price', 'country', 'city', 'property_type']

    # Vérification de l'intégrité des colonnes
    missing = [c for c in req_cols if c not in df_proc.columns]
    if missing:
        st.error(f"Colonnes manquantes dans le fichier CSV : {missing}")
        return None

    # Nettoyage des valeurs nulles
    nb_avant = len(df_proc)
    df_proc = df_proc.dropna(subset=req_cols)

    # Il faut assez de lignes pour le split Train/Test et la Cross-Validation
    nb_min = max(20, cv_folds * 5)
    if len(df_proc) < nb_min:
        cols_vides = [c for c in req_cols if df[c].isna().all()]
        detail = f" Colonnes entièrement vides : {cols_vides}." if cols_vides else ""
        st.error(f"Pas assez de données exploitables : {len(df_proc)} ligne(s) sur {nb_avant} après suppression "
                 f"des valeurs manquantes et des outliers (minimum requis : {nb_min}).{detail} "
                 f"Vérifiez le séparateur et le contenu de votre CSV.")
        return None

    # Calcul de l'âge du bien (plus pertinent que l'année de construction brute)
    current_year = datetime.datetime.now().year
    df_proc['property_age'] = current_year - df_proc['constructed_year']

    # Transformation Logarithmique de la cible (Prix) pour stabiliser la variance
    y_log = np.log1p(df_proc['price'])
    X = df_proc.drop(columns=['price', 'constructed_year'])

    # Division Train / Test (80% pour apprendre, 20% pour vérifier)
    X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

    # 2. Création du Pipeline de Pré-traitement
    # Les colonnes numériques sont standardisées (mise à l'échelle).
    # Les colonnes texte (ville, type) sont encodées en vecteurs binaires (OneHot).
    num_cols = ['property_size_sqft', 'property_age', 'rooms', 'bathrooms', 'neighbourhood_rating',
                'connectivity_score', 'garage', 'garden']
    cat_cols = ['country', 'city', 'property_type']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])

    # 3. Définition des Modèles (Le Cerveau)
    # RandomForest : Configuration Full CPU (n_jobs=-1)
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    # XGBoost : Configuration optimisée pour histogrammes (très rapide sur CPU récent)
    xgb = XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42, tree_method="hist")

    # Le VotingRegressor fait la moyenne des prédictions de RF et XGB
    ensemble = VotingRegressor(estimators=[('rf', rf), ('xgb', xgb)])

    pipeline = Pipeline([('prep', preprocessor), ('ensemble', ensemble)])

    # 4. Optimisation des Hyperparamètres (GridSearch)
    param_grid = {
        'ensemble__rf__n_estimators': rf_trees,
        'ensemble__xgb__n_estimators': xgb_n_est,
        'ensemble__xgb__learning_rate': xgb_lr,
        'ensemble__xgb__max_depth': xgb_depth
    }

    total = len(rf_trees) * len(xgb_n_est) * len(xgb_lr) * len(xgb_depth)
    progress_bar.progress(5, text=f"Optimisation de {total} modèles Hybrides en cours...")

    # Lancement de la recherche (n_jobs=1 pour éviter les conflits de threads, car les modèles sont déjà parallélisés)
    grid = GridSearchCV(pipeline, param_grid, cv=cv_folds, scoring='r2', n_jobs=1, verbose=0, return_train_score=True)
    grid.fit(X_train, y_train_log)

    # 5. Évaluation finale & Conversion inverse
    best_model = grid.best_estimator_

    # On convertit les prédictions log (exp) pour revenir à des Euros réels
    y_pred_test_real = np.expm1(best_model.predict(X_test))
    y_test_real = np.expm1(y_test_log)
    y_pred_train_real = np.expm1(best_model.predict(X_train))
    y_train_real = np.expm1(y_train_log)

    # Calcul des métriques de performance
    r2_test = r2_score(y_test_real, y_pred_test_real)
    r2_train = r2_score(y_train_real, y_pred_train_real)
    mae = mean_absolute_error(y_test_real, y_pred_test_real)

    progress_bar.progress(100, text="Entraînement terminé avec succès !")

    # Retourne un dictionnaire contenant tout le nécessaire pour l'application
    return {
        'model': best_model,
        'r2': r2_test,
        'mae': mae,
        'r2_train': r2_train,
        'best_params': grid.best_params_,
        'raw_df': df_proc
    }


# ============================================================================
# 3. INTERFACE UTILISATEUR (Streamlit)
# ============================================================================

st.title("🤖 Agent Immobilier IA - V13")

# Création des onglets principaux
tab1, tab2 = st.tabs(["🏠 Estimation & Marché", "⚙️ Configuration IA"])

# --- ONGLET 1 : ESTIMATION ---
with tab1:
    # 1. Chargement intelligent du modèle
    # On vérifie d'abord si le modèle est déjà chargé en session, sinon on le cherche sur le disque.
    if 'hybrid_model' not in st.session_state:
        if os.path.exists(MODELE_FILE):
            try:
                st.session_state['hybrid_model'] = joblib.load(MODELE_FILE)
            except:
                pass

    model_data = st.session_state.get('hybrid_model')

    # 2. Cas où aucun modèle n'est trouvé : On propose l'upload manuel
    if model_data is None:
        st.warning("⚠️ Aucun modèle détecté automatiquement.")
        st.info(
            "👇 Vous pouvez charger un cerveau existant ici OU en entraîner un nouveau dans l'onglet 'Configuration'.")

        uploaded = st.file_uploader("📥 Charger un fichier .joblib existant", type=['joblib'])
        if uploaded:
            try:
                st.session_state['hybrid_model'] = joblib.load(uploaded)
                st.success("✅ Modèle chargé avec succès ! L'interface va s'actualiser...")
                st.rerun()  # Recharge la page pour afficher le formulaire
            except Exception as e:
                st.error(f"Fichier invalide : {e}")

    # 3. Cas où le modèle est chargé : On affiche le formulaire d'estimation
    else:
        df_ref = model_data.get('raw_df')

        st.markdown("#### 📝 Saisie des caractéristiques du bien")

        # --- Formulaire ---
        c1, c2, c3 = st.columns(3)
        pays_list = sorted(df_ref['country'].unique())
        pays = c1.selectbox("Pays", pays_list)
        villes = sorted(df_ref[df_ref['country'] == pays]['city'].unique())
        ville = c2.selectbox("Ville", villes)
        types = sorted(df_ref['property_type'].unique())
        p_type = c3.selectbox("Type de bien", types)

        st.markdown("---")

        col_A, col_B, col_C = st.columns(3)
        with col_A:
            # --- MODIFICATION ICI : Conversion m² vers sqft ---
            # L'utilisateur saisit en m², on convertit pour le modèle qui a appris en sqft
            surface_m2 = st.number_input("Surface (m²)", min_value=9, max_value=5000, value=100, step=5)
            sqft = surface_m2 * 10.7639  # Conversion : 1 m² = 10.7639 sqft
            st.caption(f"≈ {sqft:.0f} sqft (unité utilisée par l'IA)")

            annee = st.number_input("Année Construction", min_value=1800, max_value=datetime.datetime.now().year, value=2010, step=1)

        with col_B:
            chambres = st.number_input("Nombre de pièces", min_value=0, max_value=50, value=3, step=1)
            sdb = st.number_input("Salles de Bain", min_value=0, max_value=20, value=2, step=1)

        with col_C:
            quartier = st.number_input("Note Quartier (0-10)", min_value=0.0, max_value=10.0, value=8.0, step=0.1)
            conn = st.number_input("Connectivité (0-10)", min_value=0.0, max_value=10.0, value=7.5, step=0.1)

        st.write("")
        col_opt1, col_opt2, _ = st.columns([1, 1, 3])
        garage = 1 if col_opt1.checkbox("Garage", value=True) else 0
        garden = 1 if col_opt2.checkbox("Jardin", value=False) else 0

        # Préparation des données pour la prédiction
        input_data = pd.DataFrame([{
            'property_size_sqft': sqft, 'property_age': datetime.datetime.now().year - annee, 'rooms': chambres,
            'bathrooms': sdb, 'neighbourhood_rating': quartier, 'connectivity_score': conn,
            'garage': garage, 'garden': garden, 'country': pays, 'city': ville, 'property_type': p_type
        }])

        st.markdown("---")

        # --- Action d'Estimation ---
        if st.button("🔮 LANCER L'ESTIMATION", type="primary"):
            pipeline = model_data['model']

            # Prédiction log -> conversion exp
            log_pred = pipeline.predict(input_data)[0]
            prix_estime = np.expm1(log_pred)

            # Affichage du Prix
            st.markdown(f"""
            <div class="metric-box">
                <h3 style="color:#e67e22; margin:0;">Estimation de Valeur</h3>
                <h1 style="color:#2c3e50; font-size:4em; margin:10px 0;">{prix_estime:,.0f} €</h1>
                <p style="color:#27ae60; font-weight:bold;">Intervalle de confiance : ± {model_data['mae']:,.0f} €</p>
            </div>
            """, unsafe_allow_html=True)

            # Affichage du Graphique de Positionnement
            st.markdown("### 📊 Positionnement sur le marché local")

            # Filtrage des données de référence pour comparaison pertinente
            subset = df_ref[df_ref['country'] == pays]
            if len(subset[subset['city'] == ville]) > 50:
                subset = subset[subset['city'] == ville]
                titre_graph = f"Distribution des prix à {ville}"
            else:
                titre_graph = f"Distribution des prix en {pays} (Données ville insuffisantes)"

            # Création de l'histogramme avec Plotly
            fig = px.histogram(subset, x="price", nbins=50, title=titre_graph, color_discrete_sequence=['#bdc3c7'])
            # Ajout de la ligne verticale représentant l'estimation
            fig.add_vline(x=prix_estime, line_width=3, line_dash="dash", line_color="#e74c3c")
            fig.add_annotation(x=prix_estime, y=0, text="Votre Bien", showarrow=True, arrowhead=1, ax=0, ay=-40,
                               bgcolor="#e74c3c", font=dict(color="white"))
            fig.update_layout(xaxis_title="Prix (€)", yaxis_title="Nombre de biens", showlegend=False)
            st.plotly_chart(fig, width="stretch")

# --- ONGLET 2 : ENTRAÎNEMENT ---
with tab2:
    st.markdown("### 🛠️ Configuration Avancée & Entraînement")

    file = st.file_uploader("Dataset CSV", type=['csv'])
    if file:
        villes, types, df_raw, n_tot, n_doub = charger_donnees_expert(file)
        if df_raw is not None:
            # Affichage des métriques de qualité des données
            c1, c2, c3 = st.columns(3)
            c1.metric("Lignes Totales", n_tot)
            c2.metric("Doublons retirés", n_doub, delta_color="inverse")
            df_clean, n_out, _, _ = filtrer_outliers(df_raw)
            c3.metric("Outliers retirés", n_out, delta_color="inverse")

            st.markdown("---")

            # Guide Pédagogique
            with st.expander("🎓 **GUIDE DE L'EXPERT : Comprendre les Hyperparamètres**", expanded=True):
                st.markdown("""
                ### 🌲 RandomForest (Stabilité)
                * **N Arbres :** Nombre d'arbres de décision dans la forêt. Plus il y en a, plus le modèle est stable. *Conseil: 100-200*.

                ### 🚀 XGBoost (Précision)
                * **N Arbres :** Nombre d'itérations de correction. XGBoost apprend de ses erreurs. *Conseil: 500-1000*.
                * **Learning Rate :** Vitesse à laquelle le modèle apprend. Un taux faible (0.05) nécessite plus d'arbres mais est plus précis.
                * **Profondeur :** Complexité de l'arbre. Trop profond = risque d'apprendre par cœur (sur-apprentissage). *Conseil: 5-7*.

                ### 💡 Conseil Performance :
                👉 **Pour un dataset > 100k lignes :** Privilégiez un *Learning Rate* faible (0.05) et une profondeur moyenne (7).
                """)

            st.subheader("⚙️ Paramètres d'Entraînement")
            c_rf, c_xgb, c_cv = st.columns(3)

            with c_rf:
                st.markdown("**🌲 RandomForest**")
                rf_trees = st.multiselect("N Arbres", [100, 200, 300], [200])

            with c_xgb:
                st.markdown("**🚀 XGBoost**")
                xgb_n_est = st.multiselect("N Arbres XGB", [500, 1000, 1500], [1000])
                xgb_lr = st.multiselect("Learning Rate", [0.01, 0.05, 0.1], [0.05])
                xgb_depth = st.multiselect("Profondeur", [5, 7, 9], [7])

            with c_cv:
                st.markdown("**🛡️ Validation**")
                cv = st.slider("Cross-Validation (Plis)", 2, 5, 3)

            # Lancement du processus
            if st.button("🚀 LANCER L'ENTRAÎNEMENT", type="primary"):
                if not rf_trees or not xgb_lr or not xgb_depth:
                    st.error("Sélectionnez les paramètres.")
                else:
                    bar = st.progress(0, text="Initialisation...")
                    res = train_hybrid_model(df_clean, bar, rf_trees, xgb_n_est, xgb_lr, xgb_depth, cv)

                    if res:
                        st.markdown("### 🏆 Résultats de performance")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("R² Test (Généralisation)", f"{res['r2']:.2%}")
                        m2.metric("R² Train (Apprentissage)", f"{res['r2_train']:.2%}")
                        m3.metric("Erreur Moyenne (MAE)", f"{res['mae']:,.0f} €")

                        # Alerte Anti-Triche (Données synthétiques)
                        if res['r2'] > 0.999:
                            st.warning(
                                "⚠️ Score suspect de 100% : Vos données semblent artificielles (générées par ordinateur).")
                        else:
                            st.success("✅ Modèle Validé et Robuste.")

                        # Sauvegarde automatique et bouton de téléchargement
                        joblib.dump(res, MODELE_FILE)
                        st.session_state['hybrid_model'] = res

                        buffer = io.BytesIO()
                        joblib.dump(res, buffer, compress=3)
                        st.download_button("📥 Télécharger le Modèle (.joblib)", buffer.getvalue(), MODELE_FILE)

st.caption("IA Immobilier V13 | Projet académique 2025")