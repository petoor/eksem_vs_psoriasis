from scipy import misc
import numpy as np
import os         
import tensorflow as tf
import matplotlib.pyplot as plt
import utils
import random
from tensorflow.contrib.layers import fully_connected, convolution2d, flatten, max_pool2d,dropout
pool = max_pool2d
conv = convolution2d
dense = fully_connected
from spatial_transformer import transformer
from tensorflow.python.ops.nn import relu

gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
BATCH_SIZE = 1
IMG_HEIGHT = 630
IMG_WIDTH = 945
NUM_COL_CHANNELS = 3
NUM_CLASSES=2
OUT_HEIGHT= IMG_HEIGHT // 3
OUT_WIDTH = IMG_WIDTH //3

def build_model(x_pl, input_width, input_height, output_dim,
                batch_size):
    
    # Setting up placeholder, this is where your data enters the graph!
    # make distributed representation of input image for localization network
    #loc_l0 = pool(x_pl, kernel_size=[3, 3], scope="localization_l0")
    #loc_l1 = conv(loc_l0, num_outputs=32, kernel_size=[5, 5], stride=[2, 2], padding="SAME", scope="localization_l1")
    loc_l2 = pool(x_pl, kernel_size=[2, 2], scope="localization_l2")
    loc_l3 = conv(loc_l2, num_outputs=32, kernel_size=[5, 5], stride=[1, 1], padding="SAME", scope="localization_l3")
    loc_l4 = pool(loc_l3, kernel_size=[2, 2], scope="localization_l4")
    loc_l5 = conv(loc_l4, num_outputs=32, kernel_size=[5, 5], stride=[1, 1], padding="SAME", scope="localization_l5")
    loc_l6_flatten = flatten(loc_l5, scope="localization_l6-flatten")
    loc_l7 = dense(loc_l6_flatten, num_outputs=55, activation_fn=relu, scope="localization_l7")
    
    # set up weights for transformation (notice we always need 6 output neurons)
    with tf.name_scope("localization"):
        W_loc_out = tf.get_variable("localization_loc-out", [55, 6], initializer=tf.constant_initializer(0.0))
        initial = np.array([[0.4, 0, 0], [0, 0.4, 0]])
        initial = initial.astype('float32')
        initial = initial.flatten()
        b_loc_out = tf.Variable(initial_value=initial, name='b-loc-out')
        loc_out = tf.matmul(loc_l7, W_loc_out) + b_loc_out


    # spatial transformer
    l_trans1 = transformer(x_pl, loc_out, out_size=(OUT_HEIGHT, OUT_WIDTH))
    l_trans1.set_shape([None, OUT_HEIGHT, OUT_WIDTH, NUM_COL_CHANNELS])

    print( "Transformer network output shape: ", l_trans1.get_shape())

    # classification network
    #Blok 1
    #conv_l11 = conv(l_trans1, num_outputs=64, kernel_size=[3, 3])
    #conv_l12 = conv(conv_l11, num_outputs=64, kernel_size=[3, 3])
    #pool_l13 = pool(conv_l12, kernel_size=[2, 2], stride=[2,2])
    #Blok 2   
    #conv_l21 = conv(pool_l13, num_outputs=128, kernel_size=[3, 3])
    #conv_l22 = conv(l_trans1, num_outputs=128, kernel_size=[3, 3])
    #pool_l23 = pool(conv_l22, kernel_size=[2, 2], stride=[2,2])
    #Blok 3    
    #conv_l31 = conv(l_trans1, num_outputs=64, kernel_size=[3, 3])
    #conv_l32 = conv(l_trans1, num_outputs=64, kernel_size=[3, 3])
    conv_l33 = conv(l_trans1, num_outputs=75, kernel_size=[3, 3])
    pool_l34 = pool(conv_l33, kernel_size=[2, 2], stride=[2,2])
    #Blok 4    
    #conv_l41 = conv(pool_l34, num_outputs=128, kernel_size=[3, 3])
    #conv_l42 = conv(pool_l34, num_outputs=128, kernel_size=[3, 3])
    conv_l43 = conv(pool_l34, num_outputs=128, kernel_size=[3, 3])
    pool_l44 = pool(conv_l43, kernel_size=[2, 2], stride=[2,2])
    #Blok 5   
    #conv_l51 = conv(pool_l44, num_outputs=256, kernel_size=[3, 3])
    #conv_l52 = conv(pool_l44, num_outputs=256, kernel_size=[3, 3])
    conv_l53 = conv(pool_l44, num_outputs=128, kernel_size=[3, 3])
    pool_l54 = pool(conv_l53, kernel_size=[2, 2], stride=[2,2])
     
    dense_flatten = flatten(pool_l54)
    dense_1 = dense(dense_flatten, num_outputs=512, activation_fn=relu)
    dropout_l4 =dropout(dense_1)
    dense_2 = dense(dropout_l4, num_outputs=256, activation_fn=relu)
    dropout_l5 =dropout(dense_2)
    logit = dense(dropout_l5, num_outputs=output_dim, activation_fn=None)
    l_out = tf.nn.softmax(logit)

    return l_out,logit,l_trans1, loc_out

tf.reset_default_graph() 

x_pl = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, NUM_COL_CHANNELS], name='input')
y_pl = tf.placeholder(tf.float32, [None, NUM_CLASSES], name='output')
lr_pl = tf.placeholder(tf.float32, shape=[], name="learning-rate")
y_from_model,y_logits,x_transform,location_out = build_model(x_pl, IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES,batch_size=BATCH_SIZE)
              
saver = tf.train.Saver()                                               
with tf.Session() as sess:
    saver.restore(sess, "./tmp/model_SGD_small_dropout_vgg16.ckpt")
    x_pred = image
    fetches_val = [y_from_model,x_transform,location_out]
    feed_dict_val = {x_pl: x_pred}
    res, x_trans,locout = sess.run(fetches=fetches_val, feed_dict=feed_dict_val)
    output_eval = res
    
    plt.imshow(np.squeeze(x_pred))
    plt.show()
    print(locout[0][0:3])
    print(locout[0][3:6])
    plt.imshow(np.abs(np.squeeze(x_trans)))
    plt.show()
    
    jj = np.squeeze(x_pred)
    misc.imsave("jj.png",jj)
    hh= np.squeeze(np.abs(x_trans))
    misc.imsave("hh.png", hh)
    
    if(output_eval[0][0] > 0.5):
        print("Prediction : Eksem")
        print(output_eval[0][0])
    else:
        print("Prediction : Psoriasis")
        print(output_eval[0][1])
