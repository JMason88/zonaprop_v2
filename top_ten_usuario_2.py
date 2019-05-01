import sqlite3
from sqlite3 import Error
import sqlite_functions.sqlite as fun_sql
import pandas as pd


if __name__ == '__main__':
    print('Creating SQlite DB in memory...')
    conn = fun_sql.create_connection()
    print(conn)
    print('Uploading Hit Aviso to SQlite Table...')
    fun_sql.copy_table_from_df(conn=conn, filepath='data/users_hitaviso.csv', table_name='tabla')
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
    GROUP BY 1,2
    ORDER BY 1,3 DESC
    ;
    """
    
    print('Executing SQL Statement...')
    c = conn.cursor()
    c.execute(sql)
    print('Fetching Results...')
    rows = c.fetchall()
    for row in range(5):
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
        if len(dict[key]) > 10:
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

    top_ten = "b4d408ac861643e406fff0df747b1498e05c9ccf a3c012c26c38a20ea276e005fb497aa87da72a31 516475402f7189f82624606bb5ed0020671fc3c7 57d98bb45d992aaa00a2426b1c71ca3b253014f6 65e6f5fb6aa6f97acaa843e931b42d951e0c9f61 5ba93bf725a78e530df4b2e1884baf1185d45ba1 ad900f00623c0f9fc17f79f3fa5fa2b9b1a35922 f1e36e94066f7fe6806a5bfd95fa82223ef6615c 64da4815303c8168b9c54c23cc568e029a979347 c79ac2a1684ae683bfa25947c05100e628ca21e2"

    submission.loc[submission.idavisos.isnull(), ['idavisos']] = top_ten
    print(len(submission[submission.idavisos.notnull()]))
    print(len(submission[submission.idavisos.isnull()]))

    submission.to_csv('salidas/submision.csv', index=False)
    fun_sql.close_connection(conn=conn)
