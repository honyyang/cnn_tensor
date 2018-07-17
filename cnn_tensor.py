#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import random
import numpy as np
import tensorflow as tf



# 神经网络类
class Network(object):
    def __init__(self,
            conv1_filter_number=6,
            conv2_filter_number=16,
            hide_dense_units=100, saved_model_dir=None):
        '''
        构造函数
        '''
        self.saved_model_dir = saved_model_dir
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        self.this_graph = tf.Graph()
        self.this_session = tf.Session(graph=self.this_graph, config=config)
        with self.this_graph.as_default():
            self.input_tensor = tf.placeholder(tf.float32, [None, 784])
            self.label_tensor = tf.placeholder(tf.float32, [None, 10])
            input_image = tf.reshape(self.input_tensor, [-1, 28, 28, 1])
            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(
                  inputs=input_image,
                  filters=conv1_filter_number,
                  kernel_size=[5, 5],
                  padding="same",
                  activation=tf.nn.relu)
            # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
            # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(
                    inputs=pool1,
                    filters=conv2_filter_number,
                    kernel_size=[5, 5],
                    padding="same",
                    activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
            # Dense Layer
            pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * conv2_filter_number])
            dense = tf.layers.dense(inputs=pool2_flat, units=hide_dense_units, activation=tf.sigmoid)
            # Logits Layer
            output_tensor = tf.layers.dense(inputs=dense, units=10, activation=tf.sigmoid)
            self.predict_tensor = output_tensor
            self.loss = tf.losses.mean_squared_error(labels=self.label_tensor,
                    predictions=self.predict_tensor)
            self.learning_rate = tf.placeholder(tf.float32, shape=())
            self.train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
            with self.this_session.as_default():
                tf.global_variables_initializer().run()

    def load(self):
        self.this_session = tf.Session(graph=tf.Graph())
        meta_graph_def = tf.saved_model.loader.load(self.this_session, 
                ['cnn_tensor_saved_model'], self.saved_model_dir)
        signature = meta_graph_def.signature_def['cnn_tensor_signature']
        self.input_tensor = self.this_session.graph.get_tensor_by_name(signature.inputs['input_tensor'].name)
        self.label_tensor = self.this_session.graph.get_tensor_by_name(signature.inputs['label_tensor'].name)
        self.learning_rate = self.this_session.graph.get_tensor_by_name(signature.inputs['learning_rate'].name)
        self.predict_tensor = self.this_session.graph.get_tensor_by_name(signature.outputs['predict_tensor'].name)

    def save(self):
        builder = tf.saved_model.builder.SavedModelBuilder(self.saved_model_dir)
        inputs = {
                'input_tensor': tf.saved_model.utils.build_tensor_info(self.input_tensor),
                'label_tensor': tf.saved_model.utils.build_tensor_info(self.label_tensor),
                'learning_rate': tf.saved_model.utils.build_tensor_info(self.learning_rate),
                }
        outputs = {
                'predict_tensor': tf.saved_model.utils.build_tensor_info(self.predict_tensor),
                }
        signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs, 'cnn_tensor_sig_name')
        with self.this_graph.as_default():
            builder.add_meta_graph_and_variables(self.this_session, 
                    ['cnn_tensor_saved_model'], 
                    {'cnn_tensor_signature':signature})
            builder.save()

    def predict(self, sample):
        '''
        使用神经网络实现预测
        sample: 输入样本
        '''
        with self.this_session.as_default():
            output = self.predict_tensor.eval({self.input_tensor: [sample]})
        return output

    def training(self, labels, data_set, rate, batch, epoch):
        '''
        训练函数
        labels: 样本标签
        data_set: 输入样本
        rate: 学习速率
        epoch: 训练轮数
        '''
        generates = range(len(labels))
        random.shuffle(generates)
        def next_batch(d):
            batch_labels = []
            batch_data_set = []
            for i in range(batch):
                batch_labels.append(labels[generates[d * batch + i]])
                batch_data_set.append(data_set[generates[d * batch + i]])
            return batch_labels, batch_data_set

        iterations = int(len(data_set) / batch)
        for i in range(epoch):
            for d in range(iterations):
                batch_labels, batch_data_set = next_batch(d)
                self.train_one_sample(batch_labels,
                        batch_data_set, rate)

    def train_one_sample(self, label, sample, rate):
        with self.this_session.as_default():
            self.train.run({self.label_tensor: label,
                self.input_tensor: sample, self.learning_rate: rate})

    def calc_gradient(self, label):
        pass

    def update_weight(self, rate):
        pass

    def dump(self):
        pass

    def calc_loss(self, label, output):
        label_tensor = tf.reshape(tf.convert_to_tensor(label, tf.float32), [-1])
        output_tensor = tf.reshape(tf.convert_to_tensor(output, tf.float32), [-1])
        loss = tf.losses.mean_squared_error(labels=label_tensor, predictions=output_tensor)
        with tf.Session():
            return loss.eval()


from input_data import get_training_data_set, show

def train_data_set():
    data_set, labels =  get_training_data_set()
    return labels, data_set

def test():
    labels, data_set = train_data_set()
    net= Network(6, 16, 100)
    rate = 0.3
    mini_batch = 3
    epoch = 10
    for i in range(epoch):
        net.training(labels, data_set, rate, mini_batch, 100)
        print(np.around(net.predict(data_set[-1]),decimals=3).reshape(10))
        print('loss: %f' % (net.calc_loss(labels[-1], net.predict(data_set[-1]))))
        rate /= 2

def predict():
    labels, data_set = train_data_set()
    print(labels[-1])
    show(data_set[-1])
    net = Network(6, 16, 100, saved_model_dir='./saved_model/mnist_100' )
    net.load()
    print(np.around(net.predict(data_set[-1]),decimals=3).reshape(10))
    print('loss: %f' % (net.calc_loss(labels[-1], net.predict(data_set[-1]))))

if __name__ == '__main__':
    test()
