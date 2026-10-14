#!/usr/bin/env bash
set -Eeuo pipefail

# Toujours revenir à la racine du projet
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "[+] Création de la venv ..."
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip wheel setuptools

# Crée un requirements minimal s'il n'existe pas
if [[ ! -f requirements.txt ]]; then
  echo "[!] requirements.txt introuvable, création d'un minimal..."
  cat > requirements.txt <<'REQ'
numpy
pandas
pyarrow
scikit-learn
xgboost
shap
matplotlib
SQLAlchemy
psycopg2-binary
pymysql
mlflow
optuna
joblib
jupyterlab
ipykernel
streamlit
black
isort
flake8
REQ
fi

echo "[+] Installation des dépendances ..."
pip install -r requirements.txt

echo "[+] Installation du kernel Jupyter ..."
python -m ipykernel install --user --name ia-immobilier --display-name "Python (ia-immobilier)"

echo "[+] Création des dossiers ..."
mkdir -p data/raw data/interim data/processed mlruns

echo "[✓] Environnement prêt. Activez-le avec: source scripts/activate.sh"

