# https://codeshare.io/adbo9e

import pandas as pd
import scipy as sp
from scipy import sparse

import sqlite3 as sql
import implicit

conn = sql.connect('datos/zonaprop.db')

print('dataset...')
interactions = pd.read_sql("SELECT idusuario, idaviso FROM contactos_train;", conn)
usuarios = interactions.idusuario.astype("category")
avisos = interactions.idaviso.astype("category")

item_user_data = sp.sparse.coo_matrix(([1.0] * interactions.shape[0], (avisos.cat.codes, usuarios.cat.codes)))

# model = implicit.als.AlternatingLeastSquares(factors=100)
model = implicit.bpr.BayesianPersonalizedRanking(factors=100)
model.fit(item_user_data)

user_items = item_user_data.T.tocsr()

## HOT
idusuarios = []
idavisos = []
for i, u in enumerate(pd.read_sql("SELECT idusuario FROM contactos_test_hot;", conn).idusuario):
    print('hot', i, "imp_bpr")

    c_u = usuarios[usuarios == u].cat.codes.unique()[0]
    preds = [p for p, _ in model.recommend(c_u, user_items, N=10, filter_already_liked_items=True)]

    idusuarios.append(u)
    idavisos.append(avisos[preds].str.cat(sep=' '))

# COLD
df_top_10 = pd.read_sql(
    "SELECT idaviso, count(*) as relevancia FROM contactos_train GROUP BY 1 ORDER BY 2 DESC LIMIT 10;", conn)
for i, u in enumerate(pd.read_sql("SELECT idusuario FROM contactos_test_cold;", conn).idusuario):
    idusuarios.append(u)
    idavisos.append(df_top_10['idaviso'].str.cat(sep=' '))
    print('cold', i, "imp_bpr")

df_salida = pd.DataFrame({'idusuario': idusuarios, 'idavisos': idavisos})
df_salida.to_csv('imp_bpr.csv', index=False)