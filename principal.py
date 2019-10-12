import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
tf.compat.v1.enable_eager_execution()
def univariate_data(dataset, indice_inicio, indice_final, tamanho_historico, target_size):
  data = []
  labels = []

  indice_inicio = indice_inicio + tamanho_historico
  if indice_final is None:
    indice_final = len(dataset) - target_size

  for i in range(indice_inicio, indice_final):
    indices = range(i-tamanho_historico, i)
    # Reshape data from (tamanho_historico,) to (tamanho_historico, 1)
    data.append(np.reshape(dataset[indices], (tamanho_historico, 1)))
    labels.append(dataset[i+target_size])
  return np.array(data), np.array(labels)

def show_plot(plot_data, delta, title):
  labels = ['Historico', 'Realidade', 'Predicao do Modelo']
  marker = ['.-', '.', 'go']
  time_steps = create_time_steps(plot_data[0].shape[0])
  if delta:
    future = delta
  else:
    future = 0

  plt.title(title)
  for i, x in enumerate(plot_data):
    if i:
      plt.plot(future, plot_data[i], marker[i], markersize=10,
               label=labels[i])
    else:
      plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
  plt.legend()
  plt.xlim([time_steps[0], (future+5)*2])
  plt.xlabel('Time-Step')
  return plt

def create_time_steps(length):
  time_steps = []
  for i in range(-length, 0, 1):
    time_steps.append(i)
  return time_steps

df = pd.read_csv('indice.csv')
# Removendo do banco a coluna Volume, cujo tipo eh object.
del df['volume']

df['data'] = pd.Categorical(df['data'])
df['data'] = df.data.cat.codes

# Separando para estudo, a coluna Último, com o valor de fechamento da bolsa no dia, e nao maximo nem minimo.

uni_data = df['ultimo']
uni_data.index = df['data']
uni_data  = uni_data.values

"""
# Normalizando os dados(Reais entre 0 e 1)
maior = uni_data.max()
uni_data = uni_data/maior

"""
# Normalizando os dados(Reais entre 0 e 1)
uni_train_mean = uni_data.mean()
uni_train_std = uni_data.std()
uni_data = (uni_data-uni_train_mean)/uni_train_std
print(uni_data.max())
print(uni_data.min())
maior = uni_train_std

x_train_uni,y_train_uni = univariate_data(uni_data, 0, len(uni_data),100,0)
x_val_uni,y_val_uni = univariate_data(uni_data, 0, len(uni_data),100,0)

# Aqui comeca aparte de Redes Neurais.

BATCH_SIZE = 256
BUFFER_SIZE = 10000

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

# Criando um modelo simples e compilando, com entrada de 8 neuronios LSTM e saida de um neuronio Dense

simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
    tf.keras.layers.Dense(80, activation='sigmoid'),
    tf.keras.layers.Dense(30, activation='tanh'),
    tf.keras.layers.Dense(1)
])

simple_lstm_model.compile(optimizer='adam', loss='mae')

# # Treinando o modelo com os dados da bovespa.
simple_lstm_model.fit(train_univariate, epochs = 4, steps_per_epoch=200,validation_data=val_univariate, validation_steps=100)
simple_lstm_model.save('modelo1.h5')

#simple_lstm_model = tf.keras.models.load_model('modelo1.h5')



for x, y in val_univariate.take(10):
  predict = simple_lstm_model.predict(x)
  x *= maior
  y *= maior
  predict *= maior
  x += uni_train_mean
  y += uni_train_mean
  predict += uni_train_mean
  plot = show_plot([x[255].numpy(), y[255].numpy(),
                predict[255]], 0, 'Modelo LSTM Simples')
  plot.show()
  media = 0.0
  for k in range(len(x)):
    media += abs(y[k].numpy() - predict[k])
  media = media / 256
  print(media)
  print("MAIOR DEMINH DE VAEFDCF: " + str(np.amax(x.numpy())))
    
