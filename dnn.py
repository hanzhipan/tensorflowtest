#!/usr/bin/python
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import urllib
import random

tf.logging.set_verbosity(tf.logging.ERROR)  # 日志级别设置成 ERROR，避免干扰
np.set_printoptions(threshold='nan')  # 打印内容不限制长度

QUADRANT_TRAINING = "quadrant_training.csv"
QUADRANT_TEST = "quadrant_test.csv"


def gen_data(file, count):
    with open(file, "w") as f:
        # 首行，写入数据集的组数和特征的数量
        f.write("%d,2n" % count)

        # 原点
        f.write("0,0,0n")

        # 产生一个随机坐标(x,y)
        for i in range(1, count):
            x = random.uniform(-10, 10)
            y = random.uniform(-10, 10)

            if abs(x) < 0.2:
                x = 0
            if abs(y) < 0.2:
                y = 0

            # 获得坐标的象限
            quadrant = 0
            if x > 0 and y > 0:
                quadrant = 1
            elif x < 0 and y > 0:
                quadrant = 2
            elif x < 0 and y < 0:
                quadrant = 3
            elif x > 0 and y < 0:
                quadrant = 4

            f.write("%f,%f,%dn" % (x, y, quadrant))


def main():
    # 生成训练集和测试集
    if not os.path.exists(QUADRANT_TRAINING):
        gen_data(QUADRANT_TRAINING, 2000)

    if not os.path.exists(QUADRANT_TEST):
        gen_data(QUADRANT_TEST, 5000)

    # 加载数据
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=QUADRANT_TRAINING,
                                                                       target_dtype=np.int, features_dtype=np.float32)

    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=QUADRANT_TEST,
                                                                   target_dtype=np.int, features_dtype=np.float32)

    # 2 维数据
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=2)]

    # 改造一个分类器
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[10, 20, 10],
                                                n_classes=5,
                                                model_dir="/tmp/quadrant_model")

    # 构造训练输入函数
    def get_train_inputs():
        x = tf.constant(training_set.data)
        y = tf.constant(training_set.target)
        return x, y

    # 训练模型
    classifier.fit(input_fn=get_train_inputs, steps=2000)

    # 构造测试输入函数
    def get_test_inputs():
        x = tf.constant(test_set.data)
        y = tf.constant(test_set.target)

        return x, y

    # 评估准确度
    print(classifier.evaluate(input_fn=get_test_inputs, steps=1))
    accuracy_score = classifier.evaluate(input_fn=get_test_inputs, steps=1)["accuracy"]
    print("nTest Accuracy: {0:f}n".format(accuracy_score))

    # 传入数据，对其进行分类
    def new_samples():
        return np.array(
            [[1, 1], [100, 100], [-1, 1], [-100, 100], [-1, -1], [-100, -100], [1, -1], [100, -100], [100, 0], [0, 100],
             [-100, 0], [0, -100], [0, 0]], dtype=np.float32)

    predictions = list(classifier.predict(input_fn=new_samples))

    print("New Samples, Class Predictions:    {}n".format(predictions))


if __name__ == "__main__":
    main()

exit(0)
