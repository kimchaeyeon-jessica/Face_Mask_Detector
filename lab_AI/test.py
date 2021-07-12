import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.array([1,2,3])
y_data = np.array([2,4,6])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

model.compile(lose=tf.keras.losses.mean_squared_error)
print(model.summary())

hist = model.fit(x_data, y_data, epochs=5000)

plt.plot(hist.history['loss'])
plt.ylabel  #응 몰라놓침 깃허브에 이건 안올리