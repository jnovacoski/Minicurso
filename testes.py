import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

mnist = tf.keras.datasets.mnist

def gen_image(arr):
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest')
    return plt


(x_train, y_train),(x_test, y_test) = mnist.load_data()
num_alea = np.random.randint(len(x_train))
im = gen_image(x_train[num_alea])
print(" O numero eh " +str(y_train[num_alea])+" e o indice eh " + str(num_alea))
im.show()
x_train, x_test = x_train / 255.0, x_test / 255.0


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
"""
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)

"""
