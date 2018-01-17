# coding:utf-8
# 下载用于训练和测试的mnist数据集的源码

import input_data  # 调用input_data
import tensorflow as tf

mnist = input_data.read_data_sets('data/', one_hot=True)
print("type of minst %s" % (type(mnist)))
print("number of train minst %d" % (mnist.train.num_examples))
print("number of test minst %d" % (mnist.test.num_examples))

# 实现回归模型
x = tf.placeholder(tf.float32, [None, 784])
print x
W = tf.Variable(tf.zeros([784, 10]))
print W
b = tf.Variable(tf.zeros([10]))
print b
y = tf.nn.softmax(tf.matmul(x, W) + b)
print y
y_ = tf.placeholder("float", [None, 10])
print y_
# 训练模型
# 计算交叉墒
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
