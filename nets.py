#-------------------------------------
# Project: DHN for biometrics
# Date: 2019.11.02
# Author: Huikai Shao
# All Rights Reserved
#-------------------------------------
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os.path
import glob
from tensorflow.python.platform import gfile
from tensorflow.python.ops import array_ops
from tensorflow.contrib.layers import *
import time


layer = 16
regularizer = tf.contrib.layers.l2_regularizer(0.0005)
#This version is the last fc code to 128 /64/256
# cancal added conv, code for 256

def encode(inputs,Training = True,Reuse = False,alpha=0.2):
    with tf.variable_scope('encode',reuse = Reuse) as scope:
        with tf.variable_scope('encode1',reuse = Reuse) as scope:
            weight1 = tf.get_variable('weight1', [3, 3, 1, 16], tf.float32, tf.glorot_uniform_initializer()) #11 is 0.7%  5 is 0.5% 8000 trains
            bias1 = tf.get_variable('bias1',[16],tf.float32,tf.zeros_initializer())
            conv1 = tf.nn.conv2d(input=inputs, filter=weight1, strides=[1, 4, 4, 1], padding='VALID', name='deconv1')
            mean, variance = tf.nn.moments(conv1, [0, 1, 2])
            net = tf.nn.batch_normalization(conv1, mean, variance, bias1, None, 1e-5)
            net = tf.maximum(alpha*net,net)             #64
            net = tf.nn.max_pool(net,[1,2,2,1],[1,1,1,1],'VALID')
            
        with tf.variable_scope('encode2',reuse = Reuse) as scope:
            weight1 = tf.get_variable('weight1',[5, 5, 16, 32],tf.float32,tf.glorot_uniform_initializer())
            bias1 = tf.get_variable('bias1',[32],tf.float32,tf.zeros_initializer())
            conv1 = tf.nn.conv2d(input=net, filter=weight1, strides=[1, 2, 2, 1], padding='SAME', name='deconv1')
            mean, variance = tf.nn.moments(conv1, [0, 1, 2])
            net = tf.nn.batch_normalization(conv1, mean, variance, bias1, None, 1e-5)
            net = tf.maximum(alpha*net,net)             #32
            net = tf.nn.max_pool(net,[1,2,2,1],[1,1,1,1],'VALID')
              
        with tf.variable_scope('encode3',reuse = Reuse) as scope:
            weight1 = tf.get_variable('weight1',[3, 3, 32, 64],tf.float32,tf.glorot_uniform_initializer())
            bias1 = tf.get_variable('bias1',[64],tf.float32,tf.zeros_initializer())
            conv1 = tf.nn.conv2d(input=net, filter=weight1, strides=[1, 1, 1, 1], padding='SAME', name='deconv1')
            #mean, variance = tf.nn.moments(conv1, [0, 1, 2])
            #net = tf.nn.batch_normalization(conv1, mean, variance, bias1, None, 1e-5)
            net = tf.nn.bias_add(conv1,bias1)
            net = tf.maximum(alpha*net,net)             #16
        
        '''    
        with tf.variable_scope('encode4',reuse = Reuse) as scope:
            weight1 = tf.get_variable('weight1',[3, 3, 128,128],tf.float32,tf.glorot_uniform_initializer())
            bias1 = tf.get_variable('bias1',[128],tf.float32,tf.zeros_initializer())
            conv1 = tf.nn.conv2d(input=net, filter=weight1, strides=[1, 1, 1, 1], padding='SAME', name='deconv1')
            #mean, variance = tf.nn.moments(conv1, [0, 1, 2])
            #net = tf.nn.batch_normalization(conv1, mean, variance, bias1, None, 1e-5)
            net = tf.nn.bias_add(conv1,bias1)
            net = tf.maximum(alpha*net,net)             #8
        '''
        with tf.variable_scope('encode5',reuse = Reuse) as scope:
            weight1 = tf.get_variable('weight1',[3, 3, 64, 128],tf.float32,tf.glorot_uniform_initializer())
            bias1 = tf.get_variable('bias1',[128],tf.float32,tf.zeros_initializer())
            conv1 = tf.nn.conv2d(input=net, filter=weight1, strides=[1, 1, 1, 1], padding='SAME', name='deconv1')
            net = tf.nn.bias_add(conv1,bias1)
            net = tf.maximum(alpha*net,net)
            net = tf.nn.max_pool(net,[1,2,2,1],[1,1,1,1],'VALID')             #3
            code = net
                      
    return net  
    
def encode6(inputs,Training = True,Reuse = False,alpha=0.2):

    with tf.variable_scope('encode6',reuse = Reuse) as scope:
            '''
            weight1 = tf.get_variable('weight1',[3, 3, 8*layer, 8*layer],tf.float32,tf.glorot_uniform_initializer())
            bias1 = tf.get_variable('bias1',[8*layer],tf.float32,tf.zeros_initializer())
            conv1 = tf.nn.conv2d(input=inputs, filter=weight1, strides=[1, 2, 2, 1], padding='VALID', name='deconv1')
            mean, variance = tf.nn.moments(conv1, [0, 1, 2])
            net = tf.nn.batch_normalization(conv1, mean, variance, bias1, None, 1e-5)
            #net = tf.nn.bias_add(conv1,bias1)
            net = tf.maximum(alpha*net,net)             #3
            '''
            code_shape = inputs.get_shape().as_list()
            weight1 = tf.get_variable('weight1',[code_shape[1],2048],tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias1 = tf.get_variable('bias1',[2048],tf.float32,initializer=tf.zeros_initializer())
            net = tf.matmul(inputs,weight1)
            #mean, variance = tf.nn.moments(net, [0, 1])
            #net = tf.nn.batch_normalization(net, mean, variance, bias1, None, 1e-5)
            net = net + bias1
            net = tf.maximum(alpha*net,net)
            #if Training: net = tf.nn.dropout(net,0.5)
            tf.add_to_collection('losses',regularizer(weight1))
            #'''
    with tf.variable_scope('encode7',reuse = Reuse) as scope:
            '''
            weight1 = tf.get_variable('weight1',[3, 3, 8*layer, 8*layer],tf.float32,tf.glorot_uniform_initializer())
            bias1 = tf.get_variable('bias1',[8*layer],tf.float32,tf.zeros_initializer())
            conv1 = tf.nn.conv2d(input=inputs, filter=weight1, strides=[1, 2, 2, 1], padding='VALID', name='deconv1')
            mean, variance = tf.nn.moments(conv1, [0, 1, 2])
            net = tf.nn.batch_normalization(conv1, mean, variance, bias1, None, 1e-5)
            #net = tf.nn.bias_add(conv1,bias1)
            net = tf.maximum(alpha*net,net)             #3
            '''
            weight1 = tf.get_variable('weight1',[2048,2048],tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias1 = tf.get_variable('bias1',[2048],tf.float32,initializer=tf.zeros_initializer())
            net = tf.matmul(net,weight1)
            #mean, variance = tf.nn.moments(net, [0,1])
            #net = tf.nn.batch_normalization(net, mean, variance, bias1, None, 1e-5)
            net = net + bias1
            net = tf.maximum(alpha*net,net)
            #if Training: net = tf.nn.dropout(net,0.5)
            tf.add_to_collection('losses',regularizer(weight1))
    return net
            #'''
def encode7(inputs,Training = True,Reuse = False,alpha=0.2):
    with tf.variable_scope('encode8',reuse = Reuse) as scope:
            weight1 = tf.get_variable('weight1',[2048,256],tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias1 = tf.get_variable('bias1',[256],tf.float32,initializer=tf.zeros_initializer())
            net = tf.matmul(inputs,weight1) + bias1
            net = tf.nn.tanh(net)
            #'''
    return net  #1


       
       
       
       
       
       
       
       
       
       
       

       
            
