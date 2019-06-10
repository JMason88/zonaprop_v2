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

def copy_table_from_df(conn, filepath, table_name, rm_duplicate = False):
    file = pd.read_csv(filepath)
    if rm_duplicate:
        file.drop_duplicates(inplace=True)
    file.to_sql(table_name, conn, index=False)

def copy_table_from_df2(conn, filepath, table_name, rm_duplicate = False):
    file = pd.read_pickle(filepath)
    if rm_duplicate:
        file.drop_duplicates(inplace=True)
    file.to_sql(table_name, conn, index=False)

def sql_to_pandas(conn, sql):
    cur = conn.cursor()
    cur.execute(sql)
    df = pd.DataFrame(cur.fetchall())
    df.columns = list(map(lambda x: x[0], cur.description))
    return df