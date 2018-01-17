#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import input_data


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w):
    return tf.matmul(X, w)


# 导入数据
mnist = input_data.read_data_sets("data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
# 设置占位符
X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])
# 初始化权重
w = init_weights([784, 10])
# 构建模型
py_x = model(X, w)
# 构建损失函数，我们采用softmax和交叉熵来训练模型
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
learning_rate = 0.01
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
predict_op = tf.argmax(py_x, 1)
with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    for i in xrange(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
    print i, np.mean(np.argmax(teY, axis=1) == sess.run(predict_op, feed_dict={X: teX, Y: teY}))
