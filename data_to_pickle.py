import pandas as pd

print('Loading train...')
train = pd.read_csv('data/contactos_train.csv')
print(train.head())
print('Saving train to pickle...')
train.to_pickle('data/contactos_train.pkl')

print('Loading test...')
test = pd.read_csv('data/contactos_test.csv')
print(test.head())
print('Saving test to pickle...')
test.to_pickle('data/test.pkl')

print('Loading User Features...')
educacion = pd.read_csv('data/users_features.csv')
print(educacion.head())
print('Saving User Features to pickle...')
educacion.to_pickle('data/user_features.pkl')

print('Loading Item Features...')
genero_edad = pd.read_csv('data/items_features.csv')
print(genero_edad.head())
print('Saving Item Features to pickle...')
genero_edad.to_pickle('data/item_features.pkl')

print('Loading Hit Avisos...')
avisos = pd.read_csv('data/users_hitaviso.csv')
print(avisos.head())
print('Saving Hit Avisos to pickle...')
avisos.to_pickle('data/hit_avisos.pkl')

