import sqlite3

conn = sqlite3.connect("predictions.db")
print(conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())
print(conn.execute("PRAGMA table_info(prediction_history)").fetchall())
conn.close()
