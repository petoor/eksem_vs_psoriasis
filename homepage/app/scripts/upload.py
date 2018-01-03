from flask import send_from_directory, send_file, render_template
from werkzeug import secure_filename
import os
import sys
from scipy import misc
import numpy as np
import PIL
import os         
import tensorflow as tf
import matplotlib.pyplot as plt
#import utils
import random
from tensorflow.contrib.layers import fully_connected, convolution2d, flatten, max_pool2d,dropout
pool = max_pool2d
conv = convolution2d
dense = fully_connected
from .spatial_transformer import transformer
from tensorflow.python.ops.nn import relu

def build_model(x_pl, input_width, input_height, output_dim,
                batch_size):
    
    # make distributed representation of input image for localization network
    loc_l1 = pool(x_pl, kernel_size=[2, 2], scope="localization_l1")
    loc_l2 = conv(loc_l1, num_outputs=8, kernel_size=[5, 5], stride=[1, 1], padding="SAME", scope="localization_l2")
    loc_l3 = pool(loc_l2, kernel_size=[2, 2], scope="localization_l3")
    loc_l4 = conv(loc_l3, num_outputs=8, kernel_size=[5, 5], stride=[1, 1], padding="SAME", scope="localization_l4")
    loc_l4_flatten = flatten(loc_l4, scope="localization_l4-flatten")
    loc_l5 = dense(loc_l4_flatten, num_outputs=50, activation_fn=relu, scope="localization_l5")
    
    # set up weights for transformation (notice we always need 6 output neurons)
    with tf.name_scope("localization"):
        W_loc_out = tf.get_variable("localization_loc-out", [50, 6], initializer=tf.constant_initializer(0.0))
        initial = np.array([[0.45, 0, 0], [0, 0.45, 0]])
        initial = initial.astype('float32')
        initial = initial.flatten()
        b_loc_out = tf.Variable(initial_value=initial, name='b-loc-out')
        loc_out = tf.matmul(loc_l5, W_loc_out) + b_loc_out


    # spatial transformer
    l_trans1 = transformer(x_pl, loc_out, out_size=(630//3, 945//3))
    l_trans1.set_shape([None, 630//3, 945//3, 3])

    print( "Transformer network output shape: ", l_trans1.get_shape())

    # classification network
    #Blok 1
    conv_l11 = conv(l_trans1, num_outputs=64, kernel_size=[3, 3])
    conv_l12 = conv(conv_l11, num_outputs=64, kernel_size=[3, 3])
    pool_l13 = pool(conv_l12, kernel_size=[2, 2], stride=[2,2])
    conv_l32 = conv(pool_l13, num_outputs=64, kernel_size=[3, 3])
    conv_l33 = conv(conv_l32, num_outputs=64, kernel_size=[3, 3])
    pool_l34 = pool(conv_l33, kernel_size=[2, 2], stride=[2,2])
    #Blok 4    
    conv_l41 = conv(pool_l34, num_outputs=128, kernel_size=[3, 3])
    conv_l42 = conv(conv_l41, num_outputs=128, kernel_size=[3, 3])
    conv_l43 = conv(conv_l42, num_outputs=128, kernel_size=[3, 3])
    pool_l44 = pool(conv_l43, kernel_size=[2, 2], stride=[2,2])
    #Blok 5   
    conv_l51 = conv(pool_l44, num_outputs=256, kernel_size=[3, 3])
    conv_l52 = conv(conv_l51, num_outputs=256, kernel_size=[3, 3])
    conv_l53 = conv(conv_l52, num_outputs=256, kernel_size=[3, 3])
    pool_l54 = pool(conv_l53, kernel_size=[2, 2], stride=[2,2])
     
    dense_flatten = flatten(pool_l54)
    dense_1 = dense(dense_flatten, num_outputs=2048, activation_fn=relu)
    dropout_l4 =dropout(dense_1)
    dense_2 = dense(dropout_l4, num_outputs=2048, activation_fn=relu)
    dropout_l5 =dropout(dense_2)
    logit = dense(dropout_l5, num_outputs=output_dim, activation_fn=None)
    l_out = tf.nn.softmax(logit)

    return l_out, logit, l_trans1, loc_out

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in set(['jpg','bmp','gif','jpeg', 'png'])

def handle_file(request, folder):
    if 'file' not in request.files:
        return 'No file uploaded', 400
    file = request.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        return 'No file', 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(folder, filename)
        file.save(filepath)

        BATCH_SIZE = 1
        IMG_HEIGHT = 630
        IMG_WIDTH = 945
        NUM_COL_CHANNELS = 3
        NUM_CLASSES=2
        OUT_HEIGHT= IMG_HEIGHT // 3
        OUT_WIDTH = IMG_WIDTH //3
        img = misc.imread(filepath)
        img = misc.imresize(img, (IMG_HEIGHT, IMG_WIDTH)) / 255
        img = img.reshape((1, IMG_HEIGHT, IMG_WIDTH,NUM_COL_CHANNELS))

        tf.reset_default_graph() 
        x_pl = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, 3], name='input')
        y_pl = tf.placeholder(tf.float32, [None, NUM_CLASSES], name='output')
        lr_pl = tf.placeholder(tf.float32, shape=[], name="learning-rate")
        y_from_model,y_logits,x_transform,location_out = build_model(x_pl, IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES,batch_size=BATCH_SIZE)
        saver = tf.train.Saver()
        with tf.Session() as sess:
        	saver.restore(sess, "./scripts/tmp/model_ADAM_stor_vgg16.ckpt")
        	x_pred = img
        	fetches_val = [y_from_model,x_transform,location_out]
        	feed_dict_val = {x_pl: x_pred}
        	res, filename_zoom,locout = sess.run(fetches=fetches_val, feed_dict=feed_dict_val)
        	output_eval = res
        	filename_zoom = misc.toimage(np.abs(np.squeeze(filename_zoom)))
        	misc.imsave(filepath+"_zoom.jpg",filename_zoom)
        	#filename_zoom.save(filepath)
        	if(output_eval[0][0] > 0.5):
        		prediction = "Eksem"
        		certainty = output_eval[0][0]
        	else:
        		prediction = "Psoriasis"
        		certainty = output_eval[0][1]
        return render_template('home.html', filename = filename, filename_zoom = str(filename)+"_zoom.jpg", prediction = prediction, certainty = certainty)
    else:
        return 'Error', 400

def show_file(filename, upload_folder):
    return send_from_directory(filename, upload_folder)
