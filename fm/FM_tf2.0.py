# -*- coding:utf-8 -*-
"""
author: byangg
datettime: 2020/1/16 15:28
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow import keras as tk

from fm.util import load_dataset

x_train, y_train, x_test, y_test = load_dataset(is_label_onhot=False)
# initialize the model
num_classes = 2
lr = 0.01
batch_size = 512
k = 8
reg_l1 = 2e-2
reg_l2 = 0
feature_length = x_train.shape[1]


class FM(tk.layers.Layer):
    def __init__(self, k, regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.regularizer = regularizer

    def build(self, input_shape):
        self.v = self.add_weight(name="V", shape=(input_shape[1].value,
                                                  self.k),
                                 initializer=tf.initializers.lecun_uniform(),
                                 regularizer=self.regularizer,
                                 dtype=tf.float32,
                                 trainable=True)
        # self.v = tf.get_variable("V", (input_shape[1].value, self.k),
        #                          initializer=tf.initializers.truncated_normal(),
        #                          dtype=tf.float32,
        #                          trainable=True)
        # self.v = tf.random_normal((input_shape[1].value, self.k))

    def call(self, input, **kwargs):
        x = input[..., tf.newaxis]
        sm_square = tf.pow(tf.reduce_sum(self.v * x, axis=1), 2)
        square_sm = tf.reduce_sum(tf.pow(self.v, 2) * tf.pow(x, 2), axis=1)
        output = 0.5 * tf.reduce_sum(sm_square + square_sm, axis=1,
                                     keepdims=True)
        return output


input = tk.Input(shape=(feature_length,))
linear = tk.layers.Dense(1, kernel_regularizer='l2')(input)
fm_output = FM(k, 'l2')(input)
output = linear + fm_output
output = tk.layers.Activation('sigmoid')(output)
model = tk.Model(input, output)

model.summary()
model.compile('adam', loss=tk.losses.binary_crossentropy)

model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          batch_size=batch_size, epochs=100,
          )

pred = model.predict(x_test)
acc = accuracy_score(y_test, pred>0.5)

print(acc)
print('done')
