import numpy as np
import pandas as pd

print("Reading data...")
hits = pd.read_csv('data/users_hitaviso.csv')

print("Grouping Top Ten...")
top_ten = hits.groupby('idaviso').count().sort_values(['portal'], ascending=False).head(10)

print("Returning results...")
top_ten = pd.DataFrame(top_ten)

print(top_ten)

prediction = top_ten

print("Loading Test...")
test = pd.read_csv('data/contactos_test.csv')

print("Making predictions...")
test['idavisos'] = prediction

print(test.head(10))

print("Saving results...")
#test.to_csv('salidas/submision.csv', index=False)
