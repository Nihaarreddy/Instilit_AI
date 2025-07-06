#%%
from sqlalchemy import create_engine
import pandas as pd

def fetch_data_sqlalchemy(host, port, database, user, password, table_name):
    conn_str = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    engine = create_engine(conn_str)
    query = f"SELECT * FROM {table_name};"
    df = pd.read_sql(query, engine)
    engine.dispose()
    return df
