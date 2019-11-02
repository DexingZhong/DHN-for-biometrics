#-------------------------------------
# Project: DHN for biometrics
# Date: 2019.11.02
# Author: Huikai Shao
# All Rights Reserved
#-------------------------------------

import glob
import os.path
import random
import numpy as np
import tensorflow as tf
import nets
import tensorflow.contrib.slim as slim
from tensorflow.python.platform import gfile
from PIL import Image
from tensorflow.python.ops import array_ops
import time

train_size = 5
test_size = 5
total_size = train_size + test_size
class_size=60

def read_image():
    train_datas = []
    train_lables = []
    test_datas = []
    test_lables = []

    # train_size + test_size must equal to numbers of one people
    roi = open('roi_xjtu_a.txt')  
    roi_path = roi.readlines()   
    i = 0
    for i, image_list in enumerate(roi_path):  
        if i % total_size < train_size:        
            train_datas.append(image_list[:-1])  
            train_lables.append(int(i / total_size))   
        elif i % total_size >= train_size:  
            test_datas.append(image_list[:-1])
            test_lables.append(int(i / total_size))
    train_lable = np.zeros([class_size * train_size, class_size], np.int64)  
    test_lable = np.zeros([class_size * test_size, class_size], np.int64)  
    
    i = 0
    for label in train_lables:
        train_lable[i][label] = 1  
        i = i + 1
    i = 0
    for label in test_lables:
        test_lable[i][label] = 1
        i = i + 1
    train_lables = train_lable.reshape([class_size * train_size * class_size])  
    test_lables = test_lable.reshape([class_size * test_size * class_size])
    return train_datas, train_lables, test_datas, test_lables 


batch_size = 600
omega_size = 100


def main():
    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)  
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        with tf.device("/cpu:0"):
            logs_train_dir = './model_saver/model.ckpt'  
            sum_accuracy = 0.0
            train_data, train_lable, test_data, test_lable = read_image()
            image_batch, label_batch = get_batch(
                train_data, train_lable, class_size * train_size, batch_size, 11500, True)   
            global_step = tf.Variable(0, trainable=False)  
            leaning_rate = tf.train.exponential_decay(
                0.01, global_step, 30, 0.96, staircase=False)  
            opt = tf.train.RMSPropOptimizer(0.0001, 0.9)
            opt1 = tf.train.RMSPropOptimizer(leaning_rate, 0.9)

            with tf.device("/gpu:0"):
                code = nets.encode(image_batch, False, False)
                code_shape = code.get_shape().as_list()
                nodes = code_shape[1] * code_shape[2] * code_shape[3]
                code_list = tf.reshape(code, [code_shape[0], nodes])
                code1 = nets.encode6(code_list, False, False)
                code2 = nets.encode7(code1, False, False)
                sign_code2 = tf.sign(code2)
                archer_code, sabor_code = tf.split(
                    code2, [omega_size, batch_size - omega_size], axis=0)
                archer_label, sabor_label = tf.split(
                    label_batch, [omega_size, batch_size - omega_size], axis=0)
                archer_num = tf.arg_max(archer_label, 1)
                archer_matrix = tf.matmul(
                    archer_code, tf.transpose(archer_code))
                sabor_matrix = tf.matmul(sabor_code, tf.transpose(sabor_code))
                archer_Similarity = tf.matmul(
                    archer_label, tf.transpose(archer_label))
                sabor_Similarity = tf.matmul(
                    archer_label, tf.transpose(sabor_label))
            archer_diag = tf.transpose(tf.reshape(tf.tile(tf.diag_part(
                archer_matrix), [omega_size]), [omega_size, omega_size]))
            archer_sabor_diag = tf.transpose(tf.reshape(tf.tile(tf.diag_part(archer_matrix), [
                                             batch_size - omega_size]), [batch_size - omega_size, omega_size]))
            sabor_diag = tf.reshape(tf.tile(tf.diag_part(sabor_matrix), [omega_size]), [
                                    omega_size, batch_size - omega_size])
            sess.run(tf.initialize_all_variables())

            with tf.device("/gpu:0"):
                archer_distance = archer_diag + \
                    tf.transpose(archer_diag) - 2 * archer_matrix
                sabor_distance = sabor_diag + archer_sabor_diag - 2 * \
                    tf.matmul(archer_code, tf.transpose(sabor_code))
                archer_loss = tf.reduce_mean(1 / 2 * archer_Similarity * archer_distance + 1 / 2 * (
                    1 - archer_Similarity) * tf.maximum(360 - archer_distance, 0))
                sabor_loss = tf.reduce_mean(1 / 2 * sabor_Similarity * sabor_distance + 1 / 2 * (
                    1 - sabor_Similarity) * tf.maximum(360 - sabor_distance, 0))
                Similarity_loss = archer_loss + sabor_loss
                zero_loss = tf.reduce_mean(
                    tf.pow(tf.subtract(sign_code2, code2), 2.0))
                J_loss = Similarity_loss + 0.5 * zero_loss
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            t_vars = tf.trainable_variables()

            with tf.control_dependencies(update_ops):
                optimizer = opt.minimize(J_loss, global_step=global_step)
            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver(max_to_keep=5)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            print('start train_bottle')
            try:
                for e in range(200000):
                    if coord.should_stop():
                        break
                    _, zero_, Similarity_ = sess.run(
                        [optimizer, zero_loss, Similarity_loss])
                    if((e) % 10 == 0):
                        print("After %d training step(s),the loss is %g,%g." %
                              (e + 1, zero_, Similarity_))
                    if((e) % 100 == 0):
                        saver.save(sess, logs_train_dir,
                                   global_step=global_step)
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                coord.request_stop()


def get_batch(image, label, label_size, batch_size, Capacity, Shuffle):

    image = tf.cast(image, tf.string)
    label = tf.convert_to_tensor(label, tf.int64)
    label = tf.reshape(label, [label_size, 60])

    input_queue = tf.train.slice_input_producer(
        [image, label], shuffle=Shuffle, capacity=Capacity)
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=1)
    image = tf.image.resize_images(image, [128, 128])
    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch(
        [image, label], batch_size=batch_size, num_threads=1, capacity=Capacity)

    label_batch = tf.cast(label_batch, tf.float32)
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch


if __name__ == '__main__':
    main()
