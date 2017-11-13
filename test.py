#coding:utf-8
#下载用于训练和测试的mnist数据集的源码

import input_data # 调用input_data
mnist = input_data.read_data_sets('data/', one_hot=True)
print("type of minst %s" % (type(mnist)))
print("number of train minst %d" % (mnist.train.num_examples))
print("number of test minst %d" % (mnist.test.num_examples))
