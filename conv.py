import tensorflow as tf
import numpy as np
from numpy import array
import glob
from PIL import Image

a_0 = tf.placeholder(tf.float32, [None, 375])
y = tf.placeholder(tf.float32)

middle = 30
w_1 = tf.Variable(tf.random_normal([375, middle]))
b_1 = tf.Variable(tf.random_normal([1, middle]))
w_2 = tf.Variable(tf.random_normal([middle, 91]))
b_2 = tf.Variable(tf.random_normal([1]))


def sigma(x):
    return tf.div(tf.constant(1.0),
                  tf.add(tf.constant(1.0), tf.exp(tf.negative(x))))


z_1 = tf.add(tf.matmul(a_0, w_1), b_1)
a_1 = sigma(z_1)
z_2 = tf.add(tf.matmul(a_1, w_2), b_2)
a_2 = sigma(z_2)


def map_func(x):
    return np.float32(x[0] / 255)


def map_conv(x):
    return np.float32(x)


def load_data():
    escada = np.int64(0)
    caixa = np.int64(1)
    tabua = np.int64(2)

    x_i = []
    y_i = []

    for item_caixa in glob.glob("data/caixa/*.png"):
        x_i.append(list(map(map_func, array(Image.open(item_caixa).getdata(), np.uint8))))
        y_i.append(caixa)

    for item_escada in glob.glob("data/escada/*.png"):
        x_i.append(list(map(map_func, array(Image.open(item_escada).getdata(), np.uint8))))
        y_i.append(escada)

    for item_tabua in glob.glob("data/taubua/*.png"):
        x_i.append(list(map(map_func, array(Image.open(item_tabua).getdata(), np.uint8))))
        y_i.append(tabua)

    return {a_0: x_i, y: y_i}


diff = tf.subtract(a_2, y)

cost = tf.multiply(diff, diff)
step = tf.train.GradientDescentOptimizer(0.1).minimize(diff)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    data = load_data()

    correct_prediction = tf.equal(tf.argmax(cost, 1), tf.cast(y, tf.int64))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    for i in range(500):
        train_accuracy = sess.run(accuracy, feed_dict=data)
        var = sess.run(step, feed_dict=data)
        print(train_accuracy)
