#!/usr/bin/python
# coding=utf-8
import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)  # 日志级别设置成 ERROR，避免干扰
np.set_printoptions(threshold='nan')  # 打印内容不限制长度

test_count = 10  # 数据集数量
param_count = 5  # 变量数
t_x = np.floor(1000 * np.random.random([test_count, param_count]), dtype=np.float32)

# 要求的值
t_w = np.floor(1000 * np.random.random([param_count, 1]), dtype=np.float32)

# 根据公式 t_y = t_x * t_w 算出值 t_y
t_y = t_x.dot(t_w)

print t_x
print t_w
print t_y

# x 是输入量，对应 t_x，用于训练输入，在训练过程中，由外部提供，因此是 placeholder 类型
x = tf.placeholder(tf.float32, shape=[test_count, param_count])
y = tf.placeholder(tf.float32, shape=[test_count, 1])

# w 是要求的各个参数的权重，是目标输出，对应 t_w
w = tf.Variable(np.zeros(param_count, dtype=np.float32).reshape((param_count, 1)), tf.float32)

curr_y = tf.matmul(x, w)  # 实际输出数据
loss = tf.reduce_sum(tf.square(t_y - curr_y))  # 损失函数，实际输出数据和训练输出数据的方差之和
optimizer = tf.train.GradientDescentOptimizer(0.00000001)
train = optimizer.minimize(loss)  # 训练的结果是使得损失函数最小

LOSS_MIN_VALUE = tf.constant(1e-5)  # 达到此精度的时候结束训练

sess = tf.Session()
sess.run(tf.global_variables_initializer())
run_count = 0
last_loss = 0
while True:
    run_count = 1
    sess.run(train, {x: t_x, y: t_y})

    curr_loss, is_ok = sess.run([loss, loss < LOSS_MIN_VALUE], {x: t_x, y: t_y})
    print "运行%d 次,loss=%s" % (run_count, curr_loss)

    if last_loss == curr_loss:
        break

    last_loss = curr_loss
    if is_ok:
        break

curr_W, curr_loss = sess.run([w, loss], {x: t_x, y: t_y})
print("t_w: %snw: %snfix_w: %snloss: %snfix_w_loss:%s" % (
t_w, curr_W, np.round(curr_W), curr_loss, np.sum(np.square(t_w - np.round(curr_W)))))

exit(0)
