import recsys as rs
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from collections import defaultdict
from top_ten import top_ten_prediction
from top_ten_usuario_recsys import top_ten_usuario_recsys
from ponpare_recsys import build_user_item_mtrx

import lightfm as lfm
from lightfm import data
from lightfm import cross_validation
from lightfm import evaluation

ui_mtx = build_user_item_mtrx(
    train_file='data/contactos_train.pkl',
    test_file='data/test.pkl',
    user_file='data/user_features.pkl',
    item_file='data/item_features.pkl'
)

print(ui_mtx)

print('Reading train pickle...')
df_train = pd.read_pickle('data/contactos_train.pkl')

print(df_train.idusuario.nunique())

print('Reading test pickle...')
df_test = pd.read_pickle('data/test.pkl')

print(df_test.idusuario.nunique())

print('Counting intersection...')

intersection = pd.merge(
    df_train,
    df_test,
    left_on='idusuario',
    right_on='idusuario',
    how='inner'
)

print(intersection.idusuario.nunique())


print('Reading User Features...')
df_user = pd.read_pickle('data/user_features.pkl')

print(df_user.idusuario.nunique())

print('Counting intersection user-train...')

df_user = df_user.groupby('idusuario').count().reset_index()
df_train = df_train.groupby('idusuario').count().reset_index()


intersection = pd.merge(
    df_train,
    df_user,
    left_on='idusuario',
    right_on='idusuario',
    how='inner'
)

print(intersection.idusuario.nunique())

print('Counting intersection user-test...')

df_test = df_test.groupby('idusuario').count().reset_index()

intersection = pd.merge(
    df_test,
    df_user,
    left_on='idusuario',
    right_on='idusuario',
    how='inner'
)

print(intersection.idusuario.nunique())

print("Check Users menos test: {}".format(df_user.idusuario.nunique() - intersection.idusuario.nunique()))

