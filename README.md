# 🏠 Agent Immobilier IA (V13.2)

> **Projet Académique 2025** | Estimation Immobilière par Intelligence Artificielle Hybride.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-green)

Ce projet est une application web permettant d'estimer le prix de biens immobiliers avec une haute précision. Il utilise une architecture **hybride** combinant deux puissants algorithmes de Machine Learning pour optimiser le compromis Biais-Variance.

---

## 🚀 Fonctionnalités Clés

* **🧠 IA Hybride :** Combinaison d'un **RandomForest** (Stabilité) et d'un **XGBoost** (Précision) via un *Voting Regressor*.
* **🛡️ Anti-Suroptimisation :** Détection automatique des doublons et alertes en cas de scores suspects (100%).
* **📊 Visualisation de Marché :** Histogramme interactif positionnant le bien estimé par rapport au marché local.
* **⚙️ Entraînement Personnalisé :** Interface complète pour ré-entraîner le modèle sur vos propres données CSV.
* **🌍 Conversion Intelligente :** Saisie ergonomique en **m²** convertie automatiquement en **sqft** pour le modèle.

## 🛠️ Installation

### Pré-requis
* Python 3.8 ou supérieur
* Un environnement virtuel (recommandé)

### Étapes d'installation

1. **Cloner le dépôt :**
```bash
   git clone [https://github.com/votre-username/agent-immobilier-ia.git](https://github.com/votre-username/agent-immobilier-ia.git)
   cd agent-immobilier-ia
```

2. **Installer les dépendances :**
```bash
pip install -r requirements.txt
```

3. **Lancer l'application :**
```bash
streamlit run app.py
```

---

## 📖 Guide d'Utilisation

### Onglet 1 : Estimation & Marché

* Si un modèle (`.joblib`) existe, il est chargé automatiquement. Sinon, vous pouvez en uploader un.
* Remplissez les caractéristiques du bien (Surface, Chambres, Quartier...).
* Cliquez sur **"Lancer l'Estimation"** pour obtenir le prix et le graphique de positionnement.

### Onglet 2 : Configuration IA

* Chargez votre fichier CSV (colonnes requises : `price`, `property_size_sqft`, `rooms`, etc.).
* Configurez les hyperparamètres (ou utilisez les réglages par défaut optimisés pour CPU Ryzen).
* Lancez l'entraînement. Le modèle sera sauvegardé automatiquement pour l'estimation.

---

## 🏗️ Architecture Technique

Le projet repose sur un pipeline Scikit-Learn robuste :

1. **Data Cleaning :** Suppression des doublons et filtrage des outliers (interquartile range).
2. **Feature Engineering :** Création de variables temporelles (`property_age`) et Log-Transformation de la cible (`price`).
3. **Preprocessing :** `OneHotEncoder` pour les variables catégorielles et `StandardScaler` pour les numériques.
4. **Modélisation :** Architecture **Ensemble Learning** (VotingRegressor).

---

## 👥 Auteurs

* **Neil LEGENDRE-FERREIRA DA COSTA** (Back-end / Data Science)
* **Rachelle ETOO AZOMBO** (Front-end / UX)
