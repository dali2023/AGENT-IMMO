# src/data/load_csv_to_sqlite.py

import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Charger les variables d’environnement
load_dotenv()

# Paramètres SQLite
sqlite_path = os.getenv("SQLITE_PATH", "./data/raw/immobilier.db")
csv_path = "./data/raw/house_data.csv"
table_name = "annonces"

def load_csv_to_sqlite():
    # Lire le CSV avec pandas
    df = pd.read_csv(csv_path)
    
    # Créer le moteur SQLite
    engine = create_engine(f"sqlite:///{sqlite_path}")
    
    # Écrire dans SQLite (écrase si existe déjà)
    df.to_sql(table_name, con=engine, if_exists="replace", index=False)
    
    print(f"✅ Table `{table_name}` importée dans {sqlite_path} avec {len(df)} lignes.")

if __name__ == "__main__":
    load_csv_to_sqlite()

