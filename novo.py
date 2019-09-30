import tensorflow as tf
import numpy as np
import pandas as pd
#from matplotlib import pyplot as plt

df = pd.read_csv('indice.csv')

del df['volume']

df['data'] = pd.Categorical(df['data'])
df['data'] = df.data.cat.codes

ultimo = df.pop('ultimo')

dataset = tf.data.Dataset.from_tensor_slices((df.values, ultimo.values))

train_dataset = dataset.shuffle(len(df)).batch(1)

def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(((5,), ()))),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
  ])

  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
  return model
model = get_compiled_model(
print(dataset)
