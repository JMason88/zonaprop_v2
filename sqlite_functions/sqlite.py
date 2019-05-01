import sqlite3
from sqlite3 import Error
import pandas as pd

def create_connection(db_file=':memory:'):
    """ create a database connection to a SQLite database """
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
    return conn

def close_connection(conn):
    conn.close()



def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)

def copy_expert_sqlite(conn, filepath, table_name):
    cur = conn.cursor()

    with open(filepath, 'r') as f:
        cur.execute('COPY {} FROM STDIN WITH CSV HEADER'.format(table_name), f)

    conn.commit()

def copy_table_from_df(conn, filepath, table_name):
    file = pd.read_csv(filepath)
    file.to_sql(table_name, conn, index=False)
