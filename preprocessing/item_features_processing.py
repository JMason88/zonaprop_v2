import pandas as pd
from sqlite_functions import sqlite as sql_fun
import scipy.sparse as sps
import numpy as np
import os

if __name__ == '__main__':
    con = sql_fun.create_connection()

    avisos = pd.read_csv('../data/items_features.csv')
    #avisos.drop_duplicates(inplace=True)
    print(avisos.columns)
    print(50 * '-')
    print(avisos.tipo_de_propiedad.unique())
    print(50 * '-')
    print(avisos.tipo_de_operacion.unique())
    print(50 * '-')
    print(avisos.barrio.unique())
    print(50 * '-')
    print(avisos.ambientes.unique())
    print(50 * '-')
    print(avisos.metros_totales.unique())
    print('Creating avisos table...')
    sql_fun.copy_table_from_df(con, '../data/items_features.csv', 'avisos', rm_duplicate=False)

    sql_1 = '''
    SELECT
        idaviso,
        tipo_de_operacion,
        barrio
    FROM
        avisos
    '''

    #    sql_fun.copy_table_from_df(con, '../data/postulaciones_train.csv', 'avisos', rm_duplicate=True)

    df = sql_fun.sql_to_pandas(con, sql_1)

    print(df.head())
    print(os.getcwd())
    df.to_pickle('entrada/item_features.pkl')
