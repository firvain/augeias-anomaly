import os

import pandas as pd
from colorama import Fore, init
from dotenv import load_dotenv
from sqlalchemy import create_engine

init(autoreset=True)
load_dotenv('.env')
POSTGRESQL_URL = os.environ.get("POSTGRESQL_URL")


def get_data_from_augeias_postgresql(table_name: str):
    if table_name:
        print(f"getting data from {table_name}")
        engine = create_engine(POSTGRESQL_URL)
        try:
            sql = f"""select * from "{table_name}" order by timestamp"""
            print(sql)
            df = pd.read_sql(sql, con=engine, index_col="timestamp")
            print(df.describe())

            return df
        except ValueError as e:
            print(e)
    else:
        print(f"{Fore.READ}Table name is empty!!!")
        pass
