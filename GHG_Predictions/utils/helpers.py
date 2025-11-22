import sqlite3
import pandas as pd
from utils.config import DATABASE_NAME

def load_db_table():
    conn = sqlite3.connect(DATABASE_NAME)
    df = pd.read_sql("SELECT * FROM ghg_data ORDER BY timestamp ASC", conn)
    conn.close()
    return df

def save_dataframe_to_db(df, table="ghg_data"):
    conn = sqlite3.connect(DATABASE_NAME)
    df.to_sql(table, conn, if_exists="append", index=False)
    conn.close()
