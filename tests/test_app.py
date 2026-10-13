"""Tests de l'application Agent Immobilier IA (appProtoFinal.py)."""

import importlib.util
import io
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "appProtoFinal.py"
MODELE_FILE = "modele_immo_v13.joblib"


@pytest.fixture(scope="session")
def app():
    """Importe le module de l'application (Streamlit tourne en mode « bare »)."""
    spec = importlib.util.spec_from_file_location("app_immo", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def dataset():
    """Petit jeu de données synthétique au format du dataset réel."""
    rng = np.random.default_rng(0)
    n = 600
    villes = {"France": ["Paris", "Lyon"], "Canada": ["Toronto", "Montreal"]}
    pays = rng.choice(list(villes), n)
    df = pd.DataFrame({
        "property_id": np.arange(n),
        "country": pays,
        "city": [rng.choice(villes[p]) for p in pays],
        "property_type": rng.choice(["Apartment", "Villa", "Studio"], n),
        "property_size_sqft": rng.integers(300, 5000, n),
        "constructed_year": rng.integers(1950, 2024, n),
        "rooms": rng.integers(1, 8, n),
        "bathrooms": rng.integers(1, 4, n),
        "garage": rng.integers(0, 2, n),
        "garden": rng.integers(0, 2, n),
        "neighbourhood_rating": rng.integers(1, 10, n),
        "connectivity_score": rng.integers(1, 10, n),
    })
    df["price"] = (df["property_size_sqft"] * 150 + df["rooms"] * 10_000
                   + rng.normal(0, 20_000, n)).clip(50_000).round()
    return df


def to_csv_buffer(df, sep=","):
    return io.BytesIO(df.to_csv(index=False, sep=sep).encode())


class Bar:
    def progress(self, *args, **kwargs):
        pass


@pytest.fixture(scope="session")
def trained(app, dataset):
    return app.train_hybrid_model(dataset, Bar(), [20], [50], [0.1], [3], 2)


# --- Chargement & nettoyage ------------------------------------------------

@pytest.mark.parametrize("sep", [",", ";"])
def test_chargement_detecte_le_separateur(app, dataset, sep):
    villes, types, df, n_init, n_doub = app.charger_donnees_expert(to_csv_buffer(dataset, sep))
    assert df is not None and len(df) == len(dataset)
    assert set(villes) == {"France", "Canada"}
    assert types == ["Apartment", "Studio", "Villa"]
    assert n_doub == 0


def test_chargement_supprime_doublons_et_normalise_texte(app, dataset):
    sale = pd.concat([dataset, dataset.head(10)])
    sale["city"] = sale["city"].str.lower() + "  "
    _, _, df, n_init, n_doub = app.charger_donnees_expert(to_csv_buffer(sale))
    assert n_init == len(dataset) + 10
    assert n_doub == 10
    assert set(df["city"]) == {"Paris", "Lyon", "Toronto", "Montreal"}


def test_filtrer_outliers_retire_les_extremes(app, dataset):
    df, n_out, q_low, q_high = app.filtrer_outliers(dataset)
    assert n_out > 0
    assert df["price"].between(q_low, q_high, inclusive="neither").all()


def test_filtrer_outliers_sans_colonne_prix(app, dataset):
    df, n_out, q_low, q_high = app.filtrer_outliers(dataset.drop(columns=["price"]))
    assert n_out == 0 and q_low is None and q_high is None


# --- Entraînement ----------------------------------------------------------

def test_entrainement_produit_un_modele_valide(trained):
    assert set(trained) >= {"model", "r2", "mae", "r2_train", "best_params", "raw_df"}
    assert trained["r2"] > 0.5
    assert trained["mae"] > 0


def test_entrainement_colonnes_manquantes(app, dataset):
    assert app.train_hybrid_model(dataset.drop(columns=["rooms"]), Bar(), [20], [50], [0.1], [3], 2) is None


def test_prediction_positive(trained):
    X = trained["raw_df"].drop(columns=["price", "constructed_year"]).head(5)
    prix = np.expm1(trained["model"].predict(X))
    assert (prix > 0).all()


# --- Interface Streamlit ---------------------------------------------------

@pytest.fixture
def in_tmp_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_ui_sans_modele_affiche_avertissement(in_tmp_dir):
    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    assert not at.exception
    assert any("Aucun modèle" in w.value for w in at.warning)


def test_ui_estimation_avec_modele(in_tmp_dir, trained):
    joblib.dump(trained, in_tmp_dir / MODELE_FILE)
    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert any("Estimation de Valeur" in m.value for m in at.markdown)
