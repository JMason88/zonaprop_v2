import recsys as rs
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from collections import defaultdict
#from top_ten_recsys import top_ten_recommender


import lightfm as lfm
from lightfm import data
from lightfm import cross_validation
from lightfm import evaluation

print('Reading train pickle...')
df_train = pd.read_pickle('data/train.pkl')
#df_train = df_train[:1500000]
df_train['rating'] = 1
print(df_train.head())
print(50 * '-')

print('Reading test pickle...')
df_test = pd.read_pickle('data/test.pkl')
#df_test = df_test[:1000]
print(df_test.head())
print(50 * '-')

print('Reading Avisos pickle...')
avisos = pd.read_pickle('data/avisos.pkl')
print(avisos.head())
print(50 * '-')

print('The train dataset has %s users and %s items, '
      'with %s interactions in the test and %s interactions in the training set.'
      % (len(df_train['idpostulante'].unique()), len(df_train['idaviso'].unique()), len(df_test), len(df_train)))
print(50 * '-')

print('Creating Interactions...')
interactions = rs.create_interaction_matrix(df_train,
                                            'idpostulante',
                                            'idaviso',
                                            'rating')

print("Interactions shape is {}".format(interactions.shape))
print(50 * '-')

print('Creating User Dictionary...')
user_dict = rs.create_user_dict(interactions=interactions)
print(user_dict)
print(50 * '-')

print('Creating Item Dictionary...')
avisos_dict = rs.create_item_dict(df=avisos,
                                  id_col='idaviso',
                                  name_col='titulo')
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

prediction = pd.DataFrame(lst, columns=['idpostulante', 'idaviso'])
prediction['idaviso'] = prediction['idaviso'].astype(int).astype('str')

print(prediction.head(30))

#test = pd.read_csv('data/ejemplo_solution.csv')

print('Merging results...')
submission = pd.merge(
    df_test[['idpostulante']],
    prediction,
    left_on='idpostulante',
    right_on='idpostulante',
    how='left'
)
print(submission[submission.idaviso.notnull()].head(15))
print(50*"-")
print("Cantidad de usuarios CON predicción: {}".format(len(submission[submission.idaviso.notnull()])))
print("Cantidad de usuarios SIN predicción: {}".format(len(submission[submission.idaviso.isnull()])))

top_ten_prediction = top_ten_recommender(
    test=submission[submission.idaviso.isnull()],
    verbose=False
)

submission = pd.concat([submission[submission.idaviso.notnull()], top_ten_prediction])
print(submission.head(15))
print("Cantidad de usuarios CON predicción: {}".format(len(submission[submission.idaviso.notnull()])))
print("Cantidad de usuarios SIN predicción: {}".format(len(submission[submission.idaviso.isnull()])))

print('Saving Results...')
submission[['idpostulante', 'idaviso']].to_csv('salidas/submission.csv', index=False)
