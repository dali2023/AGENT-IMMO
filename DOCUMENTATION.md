# 📚 Documentation Technique - Agent Immobilier IA

Ce document détaille les choix techniques, l'architecture logicielle et les méthodologies mathématiques employées dans la version V13.2 de l'application.

## 1. Pipeline de Données (ETL)

La qualité des données étant primordiale, un pipeline strict est appliqué avant tout entraînement via la fonction `charger_donnees_expert`.

### 1.1 Nettoyage et Anti-Fuite (Anti-Leakage)
* **Déduplication :** Nous avons identifié que les datasets immobiliers contiennent souvent des doublons parfaits (doublons de scraping). Ces lignes sont supprimées pour éviter que le modèle ne valide ses connaissances sur des données qu'il a déjà vues (ce qui causerait un score R² artificiel de 100%).
* **Filtrage des Outliers :** Les biens dont le prix se situe dans le 1er centile (1%) ou le 99ème centile sont exclus pour éviter de fausser la régression par des valeurs extrêmes (erreurs de saisie ou biens hors-norme).

### 1.2 Feature Engineering
* **Âge du bien :** Transformation de l'année de construction en âge relatif (`current_year - constructed_year`).
* **Log-Transformation (`np.log1p`) :** Appliquée sur la variable cible (`price`). Cela permet de :
    1.  Stabiliser la variance (les prix immobiliers s'étalent sur plusieurs ordres de grandeur).
    2.  Rendre la distribution des prix plus "normale" (Gaussienne), ce qui favorise la convergence des algorithmes.

---

## 2. Architecture du Modèle (Hybride)

Nous utilisons une approche d'**Ensemble Learning** via un `VotingRegressor`.

### 2.1 Les Composants
1.  **Random Forest (Bagging) :**
    * *Rôle :* Assurer la stabilité et réduire la variance.
    * *Configuration :* `n_jobs=-1` pour utiliser tous les cœurs CPU disponibles.
2.  **XGBoost (Boosting) :**
    * *Rôle :* Réduire le biais et capturer les relations non-linéaires complexes.
    * *Optimisation :* Utilisation de `tree_method="hist"` pour une accélération drastique sur CPU (optimisé pour les processeurs modernes type Ryzen 9).

### 2.2 Stratégie d'Entraînement
* Le modèle final est la moyenne pondérée des prédictions des deux algorithmes.
* **Optimisation :** Recherche d'hyperparamètres via `GridSearchCV` avec validation croisée (Cross-Validation).
* **Gestion des Ressources :** Pour éviter les crashs mémoire sous Windows avec de gros datasets, nous forçons l'exécution séquentielle du GridSearch (`n_jobs=1`) tout en laissant chaque modèle utiliser 100% du CPU.

---

## 3. Interface Utilisateur (Streamlit)

### 3.1 Ergonomie (UX)
* **Conversion Dynamique :** L'utilisateur saisit la surface en **mètres carrés (m²)** (standard européen), et le système convertit à la volée en **pieds carrés (sqft)** pour interroger le modèle.
* **Interactivité :** Utilisation de `st.number_input` pour la précision et de `st.checkbox` binaire pour les équipements.

### 3.2 Visualisation
* Utilisation de la bibliothèque **Plotly** pour générer des graphiques interactifs.
* Un histogramme compare la prédiction de l'IA à la distribution réelle des prix dans la ville concernée, offrant un contexte immédiat à l'utilisateur.

---

## 4. Structure du Projet

```text
/
├── app.py                  # Point d'entrée principal (Code Streamlit)
├── modele_immo_v13.joblib  # Modèle entraîné (persistance binaire)
├── requirements.txt        # Liste des dépendances Python
├── README.md               # Guide de démarrage rapide
└── DOCUMENTATION.md        # Ce fichier
```

---

### 3. Requirements

Pour que l'installation fonctionne, vous avec besoin de ceci :

```text
streamlit
pandas
numpy
scikit-learn
xgboost
joblib
plotly
```
