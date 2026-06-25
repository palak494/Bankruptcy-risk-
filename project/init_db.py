import sqlite3

conn = sqlite3.connect("predictions.db")

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
conn.close()
