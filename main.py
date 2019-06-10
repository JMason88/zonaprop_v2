import pandas as pd
import numpy as np
from lightfm.data import Dataset
import lightfm as lfm
from lightfm import LightFM
import recsys as rs
import scipy.sparse as sps
from collections import defaultdict
from top_ten_usuario_recsys import top_ten_usuario_recsys
import sys
import pickle as pkl

def load_pickle(filename):
    with open(filename, 'rb') as f:
        array = pkl.load(f)
    return array

if __name__ == "__main__":
    #users = pd.read_pickle('preprocessing/entrada/user_features.pkl')
    items = pd.read_pickle('preprocessing/entrada/item_features.pkl')
    avisos_a_predecir = load_pickle('preprocessing/entrada/avisos_a_predecir.pkl')
    train = pd.read_pickle('data/contactos_train.pkl')
    train = train[:100000]
    test  = pd.read_pickle('data/test.pkl')
    #test  = test[:100]
    #print("Users in users:%s" % (users.idpostulante.nunique()))
    print("Users in train:%s" % (train.idusuario.drop_duplicates().nunique()))
    print("Users in test:%s" % (test.idusuario.drop_duplicates().nunique()))
    print("Avisos in Items Features:%s" % (items.idaviso.nunique()))
    print("Avisos in avisos_a_predecir:%s" % avisos_a_predecir.shape)

    #print(users.head())
    #print(users.columns)

    print(50*'-')
    #u_list = pd.concat([users.idpostulante, train.idpostulante]).drop_duplicates().reset_index().idpostulante
    u_list = train.idusuario.drop_duplicates().reset_index().idusuario
    #print(u_list)

    i_list = pd.concat([train.idaviso, items.idaviso]).drop_duplicates().reset_index().idaviso

    #print(i_list)

    print(50 * '-')
    print('Building LightFM Dataset...')
    print(50 * '-')
    lfm_dataset = Dataset(user_identity_features=False, item_identity_features=False)


    lfm_dataset.fit(
        users=u_list,
        items=i_list,
        #user_features=np.concatenate((users.edad.drop_duplicates().values,
        #                              users.sexo.drop_duplicates().values),
        #                             axis=0),
        item_features=np.concatenate((items.tipo_de_operacion.drop_duplicates().values,
                                      items.barrio.drop_duplicates().values),
                                     axis=0)
    )

    print('Retrieving internal mappings and dictionaries...')
    u_map, u_feat_map, i_map, i_feat_map = lfm_dataset.mapping()

    print(50 * '-')
    print('Building Interactions...')
    print(50 * '-')
    interactions = train.groupby(['idusuario', 'idaviso']).agg('count').rename(
        columns={'fecha': 'rating'}).reset_index()
    #print(interactions.sort_values('rating', ascending=False).head())

    interactions = interactions.values

    print(interactions)

    res_interactions, res_weights = lfm_dataset.build_interactions(data=interactions)

    print(50 * '-')
    print('Building User Features...')
    print(50 * '-')

    #users_features = np.array([users.idpostulante.values, users[['edad', 'sexo']].values.tolist()],
    #                          dtype=np.object).T

    #u_feat = lfm_dataset.build_user_features(data=users_features, normalize=False)

    #print(u_feat[0:5])
    print(50 * '-')

    print('Building Item Features...')
    print(50 * '-')

    item_features = np.array([items.idaviso.values, items[['tipo_de_operacion', 'barrio']].values.tolist()],
                              dtype=np.object).T

    i_feat = lfm_dataset.build_item_features(data=item_features, normalize=False)

    print(i_feat[:5])

    print(50 * '-')

    print('Matrix factorization model...')

    model = lfm.LightFM(no_components=30,
                        loss='warp',
                        learning_schedule='adagrad')

    model.fit(interactions=res_interactions,
              sample_weight=res_weights,
              user_features=None,
              item_features=i_feat,
              epochs=50,
              verbose=True,
              num_threads=24)


    print(50 * "-")
    print('Building Recommendations...')

    users_to_predict = test.idusuario.values
    print("Cantidad de usuarios a predecir: %s" % len(users_to_predict))
    print(50*'-')
    i_keys = np.array([k for k, _ in i_map.items()])
    print(i_keys)
    print(50 * '-')
    print('Avisos a predecir sin ordenar: %s ...' % avisos_a_predecir[:10])
    print(50 * '-')
    avisos_a_predecir_sorted = i_keys[np.in1d(i_keys, avisos_a_predecir)]
    print('Avisos a predecir ordenados: %s ...' % avisos_a_predecir_sorted[:10])

    predictions = {}
    print(50*'-')
    for i, u in enumerate(users_to_predict):
        sys.stdout.write(
            "\rRecomendacion numero: " + str(i) + "/ " + str(len(users_to_predict)))
        sys.stdout.flush()
        if u not in u_map:
            pass
        else:
            #print(u_map[u])
            if u not in train.idusuario.values:
                pass
            else:
                known_items = train[train.idusuario == u].idaviso.drop_duplicates().values

            p = model.predict(u_map[u],
                              [i_map[a] for a in avisos_a_predecir_sorted],
                              #user_features=u_feat,
                              item_features=i_feat)

            prediccion_u = avisos_a_predecir_sorted[np.argsort(-p)]
            prediccion_u = prediccion_u[~np.in1d(prediccion_u, known_items)]

            predictions[u] = prediccion_u[:10]

            #u_map, u_feat_map, i_map, i_feat_map
    print('\n' + 50 * '-')

    #print(predictions)

    print('Grouping results per client...')
    lst = []

    for key in predictions:
        lst.append([key, ' '.join(predictions[key])])
        #for recommendation in predictions[key]:
        #    lst.append([key, recommendation])

    prediction = pd.DataFrame(lst, columns=['idusuario', 'idaviso'])
    prediction['idaviso'] = prediction['idaviso'].astype('str')


    print(prediction.head(30))

    print('Merging results...')
    submission = pd.merge(
        test[['idusuario']],
        prediction,
        left_on='idusuario',
        right_on='idusuario',
        how='left'
    )
    print(submission[submission.idaviso.notnull()].head(15))
    print(50 * "-")
    print("Cantidad de usuarios CON predicción: {}".format(len(submission[submission.idaviso.notnull()])))
    print("Cantidad de usuarios SIN predicción: {}".format(len(submission[submission.idaviso.isnull()])))

    top_ten_prediction = top_ten_usuario_recsys()

    print(top_ten_prediction)
    print(top_ten_prediction.columns)

    top_ten_prediction = pd.merge(
        left=submission[submission.idaviso.isnull()].reset_index(),
        right=top_ten_prediction,
        how='inner',
        on='idusuario'
    )

    print(top_ten_prediction.head())
    print(top_ten_prediction.columns)

    top_ten_prediction['idaviso'] = top_ten_prediction.idaviso_y
    top_ten_prediction = top_ten_prediction[['idusuario', 'idaviso']]
    print(50 * "-")
    print("Revisar")
    print(50 * "-")

    print(top_ten_prediction.head(15))

    submission = pd.concat([submission[submission.idaviso.notnull()], top_ten_prediction])
    print(submission.head(15))
    print("Cantidad de usuarios CON predicción: {}".format(len(submission[submission.idaviso.notnull()])))
    print("Cantidad de usuarios SIN predicción: {}".format(len(submission[submission.idaviso.isnull()])))

    print('Saving Results...')
    submission[['idusuario', 'idaviso']].to_csv('salidas/submission.csv', index=False)
