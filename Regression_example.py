#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import boston_housing
(train_X, train_Y), (test_X, test_Y) = boston_housing.load_data()

x_mean = train_X.mean(axis=0)
x_std = train_X.std(axis=0)

train_X -= x_mean
train_X /= x_std
test_X -= x_mean
test_X /= x_std

y_mean = train_Y.mean(axis=0)
y_std = train_Y.std(axis=0)

train_Y -= y_mean
train_Y /= y_std
test_Y -= y_mean
test_Y /= y_std

#model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=65, activation='relu', input_shape=(13, )),
    tf.keras.layers.Dense(units=52, activation='relu'),
    tf.keras.layers.Dense(units=39, activation='relu'),
    tf.keras.layers.Dense(units=26, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.06), loss='mse')
history = model.fit(train_X, train_Y, epochs = 30, batch_size =32, validation_split=0.25,
callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_lass')])

plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

#regression model evaluate
model.evaluate(test_X, test_Y)


pred_Y = model.predict(test_X)

plt.figure(figsize=(5, 5))
plt.plot(test_Y, pred_Y, 'r.')
plt.axis([min(test_Y), max(test_Y), min(test_Y), max(test_Y)])

plt.plot([min(test_Y), max(test_Y)], [min(test_Y), max(test_Y)], ls='--', c=".3")
plt.xlabel('test_Y')
plt.ylabel('pred_Y')

plt.show()
