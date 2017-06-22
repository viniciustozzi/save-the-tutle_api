import tensorflow as tf
import numpy as np
from numpy import array
from PIL import Image

n_nodes_hl1 = 50
n_nodes_hl2 = 50
n_nodes_hl3 = 50

n_classes = 3

x = tf.placeholder('float', [None, 375])

hidden_1_layer = {'weights': tf.Variable(tf.random_normal([375, n_nodes_hl1])),
                  'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'biases': tf.Variable(tf.random_normal([n_classes])), }

l1 = tf.add(tf.matmul(x, hidden_1_layer['weights']), hidden_1_layer['biases'])
l1 = tf.nn.relu(l1)

l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
l2 = tf.nn.relu(l2)

l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
l3 = tf.nn.relu(l3)

output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']


def map_func(x):
    return np.float32(x[0] / 255)


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    saver.restore(sess, "data/model.ckpt")
    print sess.run(tf.argmax(output, 1), {x: [map(map_func, array(Image.open("data/teste/1.png").getdata(), np.uint8))]})
    print sess.run(tf.argmax(output, 1), {x: [map(map_func, array(Image.open("data/teste/2.png").getdata(), np.uint8))]})
    print sess.run(tf.argmax(output, 1), {x: [map(map_func, array(Image.open("data/teste/3.png").getdata(), np.uint8))]})
