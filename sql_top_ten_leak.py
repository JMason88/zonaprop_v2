import sqlite_functions.sqlite as fun_sql
import sqlite3 as sql
import pandas as pd
import numpy as np
from top_ten import top_ten_prediction

if __name__ == '__main__':
    print('Creating SQlite DB in memory...')
    conn = fun_sql.create_connection()
    print(conn)
    print('Uploading User Features to SQlite Table...')
    fun_sql.copy_table_from_df(conn=conn, filepath='data/users_features.csv', table_name='tabla')
    print('Uploading Contacto to SQlite Table...')
    fun_sql.copy_table_from_df(conn=conn, filepath='data/contactos_train.csv', table_name='contacto')

    sql = """
    SELECT 
            idusuario,
            idaviso,
            count(*) AS cant_vista
    FROM tabla
    WHERE (idusuario || '-' || idaviso) NOT IN (
        SELECT
            (idusuario || '-' || idaviso) as id_comp
        FROM contacto 
        )
        AND
        (fecha_primer_uso > '2018-12-31'
        OR
        fecha_ultimo_uso  > '2018-12-31'
        )
    GROUP BY 1,2
    ORDER BY 1,3 DESC
    ;
    """

    print('Executing SQL Statement...')
    c = conn.cursor()
    c.execute(sql)
    print('Fetching Results...')
    rows = c.fetchall()
    for row in range(10):
        print(row)

    print('Reading rows...')
    from collections import defaultdict

    dict = defaultdict(list)
    for row in rows:
        dict[row[0]].append(row[1])

    print('Grouping results per client...')
    lst = []
    for key in dict:
        sub_lst = []
        if len(dict[key]) > 5:
            sub_lst = [key, ' '.join(dict[key][:10])]
        else:
            sub_lst = [key, ' '.join(dict[key])]
        lst.append(sub_lst)

    df = pd.DataFrame(lst, columns=['idusuario', 'idavisos'])

    test = pd.read_csv('data/contactos_test.csv')

    print('Merging results...')
    submission = pd.merge(
        test[['idusuario']],
        df,
        left_on='idusuario',
        right_on='idusuario',
        how='left'
    )
    print(submission[submission.idavisos.notnull()].head(30))
    print(len(submission[submission.idavisos.notnull()]))
    print(len(submission[submission.idavisos.isnull()]))

    top_ten = top_ten_prediction(input='data/user_features.pkl')

    submission.loc[submission.idavisos.isnull(), ['idavisos']] = top_ten
    print(len(submission[submission.idavisos.notnull()]))
    print(len(submission[submission.idavisos.isnull()]))

    submission.to_csv('salidas/submision.csv', index=False)
    fun_sql.close_connection(conn=conn)
