import os
import sqlite3
from dotenv import load_dotenv

load_dotenv()
db_path = os.getenv("SQLITE_PATH", "./data/raw/immobilier.db")

def list_tables(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    return [row[0] for row in cursor.fetchall()]

def describe_table(conn, table_name):
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name});")
    return cursor.fetchall()

if __name__ == "__main__":
    conn = sqlite3.connect(db_path)
    tables = list_tables(conn)
    print(f"📚 Tables dans {db_path}: {tables}")

    for table in tables:
       print(f"\n📄 Structure de la table: {table}")
       for col in describe_table(conn, table):
          print(f" - {col[1]} ({col[2]})")