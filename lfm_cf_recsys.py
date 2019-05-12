import recsys as rs
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from collections import defaultdict
from top_ten import top_ten_prediction
from top_ten_usuario_recsys import top_ten_usuario_recsys


import lightfm as lfm
from lightfm import data
from lightfm import cross_validation
from lightfm import evaluation

print('Reading train pickle...')
df_train = pd.read_pickle('data/contactos_train.pkl')
df_train = df_train[['idusuario','idaviso']]
print(df_train.head(5))
df_train = df_train[:100000]
df_train['rating'] = int(1)
#print(df_train[df_train['idusuario'] == 'bbafbc31dc6e26a8b2e46e0ed55a63ed1acbd7d6'])
#print(df_train[df_train['idaviso'] == 'bbafbc31dc6e26a8b2e46e0ed55a63ed1acbd7d6'])

print(df_train.head())
print(50 * '-')

print('Reading test pickle...')
df_test = pd.read_pickle('data/test.pkl')
#df_test = df_test[:1000]
print(df_test.head())
print(50 * '-')

print('Prueba...')
interseccion = pd.merge(
    df_train[['idusuario']],
    df_test[['idusuario']],
    left_on='idusuario',
    right_on='idusuario',
    how='inner'
)

print("Hay {} usuarios en Train.".format(len(df_train['idusuario'].unique())))
print("Hay {} usuarios en Test.".format(len(df_test['idusuario'].unique())))
print("Hay solo {} usuarios que son HOT.".format(len(interseccion['idusuario'].unique())))
print(50 * '-')

print('Reading Avisos pickle...')
avisos = pd.read_pickle('data/item_features.pkl')
avisos = avisos.drop_duplicates(subset=['idaviso'])
print(avisos.head())
print(50 * '-')

print('The train dataset has %s users and %s items, '
      'with %s interactions in the test and %s interactions in the training set.'
      % (len(df_train['idusuario'].unique()), len(df_train['idaviso'].unique()), len(df_test), len(df_train)))
print(50 * '-')

print('Creating Interactions...')
interactions = rs.create_interaction_matrix(df=df_train[['idusuario', 'idaviso', 'rating']],
                                            user_col='idusuario',
                                            item_col='idaviso',
                                            rating_col='rating')

print("Interactions shape is {}".format(interactions.shape))
print(50 * '-')

print('Creating User Dictionary...')
user_dict = rs.create_user_dict(interactions=interactions)
#print(user_dict)
print(50 * '-')

print('Creating Item Dictionary...')
avisos_dict = rs.create_item_dict(df=df_train,
                                  id_col='idaviso',
                                  name_col='idaviso')
# print(movies_dict)
print(50 * '-')

print('Matrix factorization model...')
mf_model = rs.runMF(interactions=interactions,
                    n_components=30,
                    loss='warp',
                    k=15,
                    epoch=50,
                    n_jobs=4)
print(50 * "-")
print('Building Recommendations...')

users = user_dict.keys()
dict = defaultdict(list)

for user in users:
    rec_list = rs.sample_recommendation_user(model=mf_model,
                                             interactions=interactions,
                                             user_id=user,
                                             user_dict=user_dict,
                                             item_dict=avisos_dict,
                                             threshold=0,
                                             nrec_items=10,
                                             show=False)
    dict[user].append(rec_list)

#print(dict)

print('Grouping results per client...')
lst = []
for key in dict:
    for recommendation in dict[key][0]:
        lst.append([key, recommendation])

print(lst)

prediction = pd.DataFrame(lst, columns=['idusuario', 'idavisos'])
prediction['idavisos'] = prediction['idavisos'].astype('str')

print(prediction.head(30))

#test = pd.read_csv('data/ejemplo_solution.csv')

print('Merging results...')
submission = pd.merge(
    df_test[['idusuario']],
    prediction,
    left_on='idusuario',
    right_on='idusuario',
    how='left'
)
print(submission[submission.idavisos.notnull()].head(15))
submission_hot = submission[submission.idavisos.notnull()]
submission_cold = submission[submission.idavisos.isnull()]
print(50*"-")
submission_hot = submission_hot.groupby(['idusuario'])['idavisos'].apply(list).reset_index()
submission_hot['idavisos'] = submission_hot['idavisos'].apply(lambda x: " ".join(x))
print(submission_hot.head(15))
print(50*"-")
print("Cantidad de usuarios CON predicción: {}".format(len(submission_hot)))
print("Cantidad de usuarios SIN predicción: {}".format(len(submission_cold)))

top_ten_prediction = top_ten_usuario_recsys()

submission_cold = pd.merge(
    submission_cold[['idusuario']],
    top_ten_prediction,
    left_on='idusuario',
    right_on='idusuario',
    how='left'
)
#submission_cold.loc[:, ['idaviso']] = top_ten_prediction

submission = pd.concat([submission_hot, submission_cold])
print(submission.head(15))
print("Cantidad de usuarios CON predicción: {}".format(len(submission[submission.idavisos.notnull()])))
print("Cantidad de usuarios SIN predicción: {}".format(len(submission[submission.idavisos.isnull()])))

print('Saving Results...')
print(submission[['idusuario', 'idavisos']].head(30))
submission[['idusuario', 'idavisos']].to_csv('salidas/submision.csv', index=False)
