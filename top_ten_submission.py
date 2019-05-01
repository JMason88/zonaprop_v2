import numpy as np
import pandas as pd

print("Reading data...")
hits = pd.read_pickle('data/contactos_train.pkl')

print("Grouping Top Ten...")
top_ten = hits.groupby('idaviso').count().sort_values(['idusuario'], ascending=False).head(10)

print("Returning results...")
top_ten = top_ten.reset_index().idaviso

print(top_ten)

prediction = ' '.join(top_ten)

print(prediction)

print("Loading Test...")
test = pd.read_pickle('data/test.pkl')

print("Making predictions...")
test['idavisos'] = prediction

print(test.head(10))

print("Saving results...")
test.to_csv('salidas/submision.csv', index=False)
