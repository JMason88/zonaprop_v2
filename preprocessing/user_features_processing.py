import os
os.chdir("../")
print(os.getcwd())

import sys
sys.path.append(os.getcwd())

import pandas as pd
import sqlite_functions.sqlite as sql_fun
import scipy.sparse as sps
import numpy as np


if __name__ == '__main__':
    con = sql_fun.create_connection()

    print(os.getcwd() +'/data/postulantes_genero_edad.csv')
    genero = pd.read_csv(os.getcwd() +'/data/postulantes_genero_edad.csv')
    genero.drop_duplicates(inplace=True)
    print(genero.fechanacimiento.unique())
    print(genero.sexo.unique())
    print('Creating genero_edad table...')
    sql_fun.copy_table_from_df(con, os.getcwd() + '/data/postulantes_genero_edad.csv', 'genero_edad', rm_duplicate=True)
    print('Creating educacion table...')
    sql_fun.copy_table_from_df(con, os.getcwd() + '/data/postulantes_educacion.csv', 'educacion')

    sql_1 = '''
    SELECT
        idpostulante,
        CASE 
            WHEN nombre = 'Posgrado' AND estado = 'En Curso' THEN "posgrado_curso"
            WHEN nombre = 'Posgrado' AND estado = 'Graduado' THEN "posgrado_graduado"
            WHEN nombre = 'Posgrado' AND estado = 'Abandonado' THEN "posgrado_abandonado"
            WHEN nombre = 'Universitario' AND estado = 'En Curso' THEN "universitario_curso"
            WHEN nombre = 'Universitario' AND estado = 'Graduado' THEN "universitario_graduado"
            WHEN nombre = 'Universitario' AND estado = 'Abandonado' THEN "universitario_abandonado"
            WHEN nombre = 'Master' AND estado = 'En Curso' THEN "master_curso"
            WHEN nombre = 'Master' AND estado = 'Graduado' THEN "master_graduado"
            WHEN nombre = 'Master' AND estado = 'Abandonado' THEN "master_abandonado"
            WHEN nombre = 'Otro' AND estado = 'En Curso' THEN "otro_curso"
            WHEN nombre = 'Otro' AND estado = 'Graduado' THEN "otro_graduado"
            WHEN nombre = 'Otro' AND estado = 'Abandonado' THEN "otro_abandonado"
            WHEN nombre = 'Terciario/Técnico' AND estado = 'En Curso' THEN "terciario_curso"
            WHEN nombre = 'Terciario/Técnico' AND estado = 'Graduado' THEN "terciario_graduado"
            WHEN nombre = 'Terciario/Técnico' AND estado = 'Abandonado' THEN "terciario_abandonado"
            WHEN nombre = 'Doctorado' AND estado = 'En Curso' THEN "doctorado_curso"
            WHEN nombre = 'Doctorado' AND estado = 'Graduado' THEN "doctorado_graduado"
            WHEN nombre = 'Doctorado' AND estado = 'Abandonado' THEN "doctorado_abandonado"
            WHEN nombre = 'Secundario' AND estado = 'En Curso' THEN "secundario_curso"
            WHEN nombre = 'Secundario' AND estado = 'Graduado' THEN "secundario_graduado"
            WHEN nombre = 'Secundario' AND estado = 'Abandonado' THEN "secundario_abandonado"
        END as educacion
    FROM
        educacion
    GROUP BY 1
    '''

    sql_3 = '''
    SELECT
        idpostulante,
        CASE
            WHEN edad < 20 THEN "<20"
            WHEN edad >= 20 AND edad < 30 THEN "20-30"
            WHEN edad >= 30 AND edad < 40 THEN "30-40"
            WHEN edad >= 40 AND edad < 50 THEN "40-50"
            WHEN edad >= 50 AND edad < 60 THEN "50-60"
            WHEN edad >= 60 AND edad < 70 THEN "60-70"
            ELSE ">70"
        END as edad,
        sexo
        FROM(    
            SELECT
                idpostulante,
                strftime('%Y', date('now')) - strftime('%Y', fechanacimiento) as edad,
                sexo
               -- sum(CASE WHEN sexo = 'MASC' THEN 1 ELSE 0 END) as hombre,
               -- sum(CASE WHEN sexo = 'FEM' THEN 1 ELSE 0 END) as mujer,
               -- sum(CASE WHEN sexo = 'NO_DECLARA' OR sexo = '0.0' THEN 1 ELSE 0 END) as otro
            FROM
                genero_edad
            GROUP BY 1)
        GROUP BY 1
    
    '''

    sql = '''
        SELECT
            b.*,
            a.educacion
        FROM
            ({}) a
            INNER JOIN
            ({}) b
            ON a.idpostulante = b.idpostulante
        ORDER BY b.idpostulante
        '''.format(sql_1, sql_3)


    df = sql_fun.sql_to_pandas(con, sql)

    df.to_pickle(os.getcwd() + '/preprocessing/entrada/user_features.pkl')
