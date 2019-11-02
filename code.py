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
import tensorflow.contrib.slim.python.slim.nets.vgg as vgg
from tensorflow.python.platform import gfile
from PIL import Image
from tensorflow.python.ops import array_ops
import time

def read_image():

    tatol_datas = []

    roi = open('roi_xjtu_a.txt')
    roi_path = roi.readlines()
    total_data = []
    i = 0
    for i,image_list in enumerate(roi_path):
            tatol_datas.append(image_list[:-1])

    return tatol_datas
results = []
def main():
    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        with tf.device("/cpu:0"):
            global_step = tf.Variable(0,trainable = False)
            total_data = read_image()
            total_image = get_batch(total_data,1,6000,False)
            #with tf.device("/gpu:0"):
            code = nets.encode(total_image,False,False)
            code_shape = code.get_shape().as_list()
            nodes = code_shape[1]*code_shape[2]*code_shape[3]
            code_list = tf.reshape(code,[code_shape[0],nodes])
            code = nets.encode6(code_list,False,False)
            code = nets.encode7(code,False,False)
            #t_vars = tf.trainable_variables()
            #r1_vars = [var for var in t_vars if 'encode' in var.name]
            
            saver = tf.train.Saver()
            sess = tf.Session()
            logs_train_dir = "./model_saver"
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            saver.restore(sess,ckpt.model_checkpoint_path)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            roi = open('roi_xjtu_a.txt')
            code_path = roi.readlines()
            print('start train_bottle')
            try:
                for code_list in code_path:
                    if coord.should_stop():
                        break
                    code_val = sess.run(code)
                    code_result = np.reshape(code_val,[1,256])
                    code_result = np.sign(code_result)
                    code_list = code_list[:-4] + 'txt'
                    np.savetxt(code_list,code_result,fmt = '%d')
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                coord.request_stop()
                           
            
def get_batch(image,batch_size, Capacity,Shuffle):

    image = tf.cast(image, tf.string)
    
    input_queue = tf.train.slice_input_producer([image],shuffle = Shuffle)
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=1)
    image = tf.image.resize_images(image, [128, 128])
    #image = tf.image.per_image_standardization(image)
  
    image_batch = tf.train.batch([image],batch_size= batch_size,num_threads= 1, 
                                                capacity = Capacity)

    image_batch = tf.cast(image_batch, tf.float32)
    
    return image_batch

if __name__ == '__main__':
    main()
