from PIL import Image
from numpy import array
import numpy as np
import glob
from json import load
import tensorflow as tf


def map_func(x):
    return np.float32(x[0] / 255)


def load_feed(x, y, keep_prob, dropout):
    escada = np.int64(0)
    caixa = np.int64(1)
    tabua = np.int64(2)
    barco = np.int64(3)
    x_img = np.int64(4)

    x_i = []
    y_i = []

    for item_caixa in glob.glob("data/71x40/caixa/*.png"):
        x_i.append(list(map(map_func, array(Image.open(item_caixa).getdata(), np.uint8))))
        y_i.append(caixa)

    for item_escada in glob.glob("data/71x40/escada/*.png"):
        x_i.append(list(map(map_func, array(Image.open(item_escada).getdata(), np.uint8))))
        y_i.append(escada)

    for item_tabua in glob.glob("data/71x40/taubua/*.png"):
        x_i.append(list(map(map_func, array(Image.open(item_tabua).getdata(), np.uint8))))
        y_i.append(tabua)

    for item_tabua in glob.glob("data/71x40/barco/*.png"):
        x_i.append(list(map(map_func, array(Image.open(item_tabua).getdata(), np.uint8))))
        y_i.append(barco)

    for item_tabua in glob.glob("data/71x40/x/*.png"):
        x_i.append(list(map(map_func, array(Image.open(item_tabua).getdata(), np.uint8))))
        y_i.append(x_img)

    for item_json_x in glob.glob("data/71x40/x/*.json"):
        x_i.append(np.asarray(load(open(item_json_x, 'r')), np.float32))
        y_i.append(x_img)

    for item_json_x in glob.glob("data/71x40/x/*.json"):
        x_i.append(np.asarray(load(open(item_json_x, 'r')), np.float32))
        y_i.append(x_img)

    for item_json_x in glob.glob("data/71x40/barco/*.json"):
        x_i.append(np.asarray(load(open(item_json_x, 'r')), np.float32))
        y_i.append(barco)

    for item_json_x in glob.glob("data/71x40/escada/*.json"):
        x_i.append(np.asarray(load(open(item_json_x, 'r')), np.float32))
        y_i.append(escada)

    for item_json_x in glob.glob("data/71x40/taubua/*.json"):
        x_i.append(np.asarray(load(open(item_json_x, 'r')), np.float32))
        y_i.append(tabua)

    return {x: x_i, y: y_i, keep_prob: dropout}


def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 71, 40, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out
