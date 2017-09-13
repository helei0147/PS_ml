import tensorflow as tf
import numpy as np
import glob

from datetime import datetime
from scipy import misc

import mnist


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint_dir', 'log/fully_connected_feed/',
                           """Directory where to read model checkpoints.""")



def eval_once(predict_op, image):

    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('_')[-1]
            print('Model restored.')
        else:
            print('No checkpoint file found')
            return

    image_outputs = []
    for i in range(3):
        feed_dict = {image_filename_placeholder: image[i*128:(i+1)*128]}
        image_outputs.append(sess.run(predict_op, feed_dict=feed_dict))


    return image_outputs
def calculate_normal_error_in_degree(est_normals, gts):
    '''
    calculate normal error in degree
    est_normals: the normals to calculate, shape = n*3
    gts: ground truth ,normals to compared to, shape = n*3
    '''
    num1 = est_normals.shape[0]
    gts = gts[:num1,...]
    to_acos = np.sum(est_normals*gts, axis = 1)
    to_acos[to_acos>1] = 1
    rad_error = np.arccos(to_acos)
    degree_error = rad_error/3.1415926*180
    return degree_error

def expand(normals, axis = 0):
    num = normals.shape[0]
    return np.concatenate(normals[np.arange(num), ...],axis)

if __name__ == '__main__':
    with tf.Graph().as_default():
         
        image_channel1 = np.load('data/test/test_channel1.npy')
        test_pixel_num = image_channel1.shape[0]
        BATCH_SIZE = 1000
        TEST_BATCH_NUM = test_pixel_num//BATCH_SIZE
        OBSERVATION_NUM = 96
        image_placeholder = tf.placeholder(tf.float32,shape = (BATCH_SIZE,OBSERVATION_NUM))
        logits, keep_prob = mnist.inference(image_placeholder)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('_')[-1]
                print('Model restored.')
            else:
                print('No checkpoint file found')

            normal_outputs = np.ndarray((TEST_BATCH_NUM,BATCH_SIZE,3))

            for i in range(TEST_BATCH_NUM):
                feed_dict = {image_placeholder: image_channel1[i*BATCH_SIZE:(i+1)*BATCH_SIZE], keep_prob: 1}
                outputs = sess.run(logits, feed_dict=feed_dict)
                normal_outputs[i,...] = outputs
    np.save('normal_outputs.npy', normal_outputs)
    
    predict_outputs = expand(normal_outputs)
    print(predict_outputs.shape)
    gts = np.load('data/test/test_normals.npy')
    degree_error = calculate_normal_error_in_degree(predict_outputs,gts)
    avg_error = np.sum(degree_error)/degree_error.shape[0]
    print('avg_error = %s' % (avg_error))
