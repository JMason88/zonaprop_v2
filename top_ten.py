import numpy as numpy
import pandas as pd

def top_ten_prediction(input = 'data/hit_avisos.pkl'):
    #print("Reading data...")
    hits = pd.read_pickle(input)

    #print("Grouping Top Ten...")
    top_ten = hits.groupby('idaviso').count().sort_values(['idusuario'], ascending=False).head(10)

    #print("Returning results...")
    top_ten = top_ten.reset_index().idaviso
    return ' '.join(top_ten)
