# tensorflow_study

## 말 그대로 군대에서 텐서플로우 공부할거임

### 챕터1 신경망 네트워크 공부


import numpy as np
import tensorflow as tf
import math


def sigmoid(x):
    return 1 / 1 + exp(-x)

x = np.array([[1,1], [1,0], [0,1], [0,0]])
y = np.array([[0], [1], [1], [0]])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=2, activation = 'sigmoid', input_shape = (2,)),
    tf.keras.layers.Dense(units=1, activation = 'sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1), loss='mse')

model.summary()

history = model.fit(x, y, epochs=2000, batch_size=1)

### 챕터2 선형회귀
