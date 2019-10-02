import tensorflow as tf
import numpy as np
import pandas as pd
#from matplotlib import pyplot as plt

df = pd.read_csv('indice.csv')

del df['volume']

df['data'] = pd.Categorical(df['data'])
df['data'] = df.data.cat.codes

uni_data = df['ultimo']
uni_data.index = df['data']

uni_data  = uni_data.values

uni_train_mean = uni_data.mean()
uni_train_std = uni_data.std()


uni_data = (uni_data-uni_train_mean)/uni_train_std
print(uni_data)
print("Tamanho " + str(len(uni_data)))
