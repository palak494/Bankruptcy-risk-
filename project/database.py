import os
import sqlite3
import uuid
import pandas as pd
from datetime import datetime

class DatabaseHandler:
    def __init__(self, db_path="predictions.db"):
        self._keep_alive = None
        if db_path == ":memory:":
            # Use SQLite shared-cache URI with a unique UUID to ensure isolation between database handler instances
            self.db_path = f"file:{uuid.uuid4()}?mode=memory&cache=shared"
            self.uri = True
            # Open a persistent connection to keep this specific in-memory database alive
            self._keep_alive = sqlite3.connect(self.db_path, uri=True)
        else:
            if not os.path.isabs(db_path):
                current_dir = os.path.dirname(os.path.abspath(__file__))
                db_path = os.path.join(current_dir, db_path)
            self.db_path = db_path
            self.uri = False
            
        self.init_db()

    def get_connection(self):
        if self.uri:
            return sqlite3.connect(self.db_path, uri=True)
        return sqlite3.connect(self.db_path)

    def init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                risk_score REAL,
                risk_level TEXT,
                top_feature_1 TEXT,
                top_feature_2 TEXT,
                top_feature_3 TEXT
            )
            """)
            conn.commit()

    def insert_prediction(self, risk_score: float, risk_level: str, top_features: list):
        # Pad top_features list to ensure we have at least 3 elements
        padded_features = list(top_features)
        while len(padded_features) < 3:
            padded_features.append(None)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO prediction_history(
                timestamp,
                risk_score,
                risk_level,
                top_feature_1,
                top_feature_2,
                top_feature_3
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                timestamp,
                risk_score,
                risk_level,
                padded_features[0],
                padded_features[1],
                padded_features[2]
            ))
            conn.commit()

    def get_history(self) -> pd.DataFrame:
        with self.get_connection() as conn:
            df = pd.read_sql_query(
                """
                SELECT *
                FROM prediction_history
                ORDER BY id DESC
                """,
                conn
            )
            return df
