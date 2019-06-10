import numpy as np
import pandas as pd
from sqlite_functions import sqlite as sql_fun
import pickle as pkl
import os


if __name__ == '__main__':
    con = sql_fun.create_connection()

    interactions = pd.read_csv('../data/contactos_train.csv')
    print(interactions.columns)
    print(50*'-')
    print(interactions.head())
    print(50*'-')
    print(interactions.fecha.max())
    print(interactions.fecha.min())
    print(50*'-')


    print('Creating interactions table...')
    sql_fun.copy_table_from_df(con, '../data/contactos_train.csv', 'interaccion', rm_duplicate=False)

    sql_1 = '''
    SELECT
        idaviso,
        count(*) as hits
    FROM
        interaccion
    --WHERE
        --fechapostulacion >= '2018-03-20'
    GROUP BY 1
    ORDER BY 2 DESC
    LIMIT 150;
    '''

    df = sql_fun.sql_to_pandas(con, sql_1)

    print(df.head())
    print(os.getcwd())
    avisos_a_predecir = df.idaviso.values
    print(avisos_a_predecir)

    with open('entrada/avisos_a_predecir.pkl', 'wb') as f:
        pkl.dump(avisos_a_predecir, f)
    #df.to_pickle('entrada/avisos_a_predecir.pkl')
