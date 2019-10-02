import tensorflow as tf
import numpy as np
import pandas as pd
#from matplotlib import pyplot as plt

df = pd.read_csv('indice.csv')
# Removendo do banco a coluna Volume, cujo tipo eh object.
del df['volume']

df['data'] = pd.Categorical(df['data'])
df['data'] = df.data.cat.codes

# Separando para estudo, a coluna Último, com o valor de fechamento da bolsa no dia, e nao maximo nem minimo.

uni_data = df['ultimo']
uni_data.index = df['data']
uni_data  = uni_data.values

# Normalizando os dados

uni_train_mean = uni_data.mean()
uni_train_std = uni_data.std()
uni_data = (uni_data-uni_train_mean)/uni_train_std

""" Testes
print(uni_data)
print("Tamanho " + str(len(uni_data)))

def retorna_dados(dataset):
  dados = []
  rotulo = []

"""
dados = []

for i in range(0, 30):
    indices = range(0,50)
    dados.append(np.reshape(df[indices], (20,1)))

print(dados)

