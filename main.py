import tensorflow as tf
from PIL import Image
from numpy import array
import numpy as np
import glob

n_nodes_hl1 = 50
n_nodes_hl2 = 50
n_nodes_hl3 = 50

n_classes = 3

x = tf.placeholder('float', [None, 375])
y = tf.placeholder(tf.int64)

hidden_1_layer = {'weights': tf.Variable(tf.random_normal([375, n_nodes_hl1])),
                  'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'biases': tf.Variable(tf.random_normal([n_classes])), }

l1 = tf.add(tf.matmul(x, hidden_1_layer['weights']), hidden_1_layer['biases'])
tf.summary.histogram("l1", l1)
l1 = tf.nn.relu(l1)
tf.summary.histogram("act", l1)

l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
l2 = tf.nn.relu(l2)

l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
l3 = tf.nn.relu(l3)

output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=y))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)


def map_func(x):
    return np.float32(x[0] / 255)


def load_data():
    escada = np.int64(0)
    caixa = np.int64(1)
    tabua = np.int64(2)

    x_i = []
    y_i = []

    for item_caixa in glob.glob("data/caixa/*.png"):
        x_i.append(map(map_func, array(Image.open(item_caixa).getdata(), np.uint8)))
        y_i.append(caixa)

    for item_escada in glob.glob("data/escada/*.png"):
        x_i.append(map(map_func, array(Image.open(item_escada).getdata(), np.uint8)))
        y_i.append(escada)

    for item_tabua in glob.glob("data/taubua/*.png"):
        x_i.append(map(map_func, array(Image.open(item_tabua).getdata(), np.uint8)))
        y_i.append(tabua)

    return {x: x_i, y: y_i}


data = load_data()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess.run(init)

    correct_prediction = tf.equal(tf.argmax(output, 1), y)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    for i in range(10000):
        train_accuracy = sess.run(accuracy, feed_dict=data)
        sess.run(train_step, feed_dict=data)
        print('Step {:5d}: training accuracy {:g}'.format(i, train_accuracy))
        writer = tf.summary.FileWriter("1")
        writer.add_graph(sess.graph)
        if train_accuracy == 1.0:
            var = sess.run(output, feed_dict=data)
            print var
            print len(var)
            break

    save_path = saver.save(sess, "data/model.ckpt")
