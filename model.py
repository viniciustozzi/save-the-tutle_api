from PIL import Image
from numpy import array
import numpy as np
import glob
from json import load
import tensorflow as tf


def map_func(x):
    return np.float32(x[0] / 255)


def map_conv(x):
    return np.float32(x)


def load_data(x, y):
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

    for item_json_x in glob.glob("data/71x40/caixa/*.json"):
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

    return {x: x_i, y: y_i}


def load_model():
    n_nodes_hl1 = 500
    n_nodes_hl2 = 500
    n_nodes_hl3 = 500
    n_nodes_hl4 = 500

    n_classes = 5

    x = tf.placeholder('float', [None, 2840])
    y = tf.placeholder(tf.int64)

    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([2840, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    hidden_4_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl4]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl4, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes])), }

    l1 = tf.add(tf.matmul(x, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    l4 = tf.add(tf.matmul(l3, hidden_4_layer['weights']), hidden_4_layer['biases'])
    l4 = tf.nn.relu(l4)

    output = tf.matmul(l4, output_layer['weights']) + output_layer['biases']

    return x, y, output
