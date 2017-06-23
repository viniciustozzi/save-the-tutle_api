import tensorflow as tf
from PIL import Image
from numpy import array
import numpy as np
import glob
from json import load

n_nodes_hl1 = 50
n_nodes_hl2 = 50
n_nodes_hl3 = 50
n_nodes_hl4 = 50

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
tf.summary.histogram("normal/l1", l1)
l1 = tf.nn.relu(l1)

l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
l2 = tf.nn.relu(l2)
tf.summary.histogram("normal/l2", l2)

l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
l3 = tf.nn.relu(l3)
tf.summary.histogram("normal/l3", l3)

l4 = tf.add(tf.matmul(l3, hidden_4_layer['weights']), hidden_4_layer['biases'])
l4 = tf.nn.relu(l4)
tf.summary.histogram("normal/l4", l4)

output = tf.matmul(l4, output_layer['weights']) + output_layer['biases']

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=y))
train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

tf.summary.histogram("normal/output", output)
tf.summary.histogram("normal/loss", loss)

summaries = tf.summary.merge_all()


def map_func(x):
    return np.float32(x[0] / 255)


def load_data():
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


data = load_data()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess.run(init)

    correct_prediction = tf.equal(tf.argmax(output, 1), y)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.histogram("normal/accuracy", accuracy)
    try:
        for i in range(1500):
            train_accuracy = sess.run(accuracy, feed_dict=data)
            sess.run(train_step, feed_dict=data)
            print('Step {:5d}: training accuracy {:g}'.format(i, train_accuracy))

            writer = tf.summary.FileWriter("1")
            writer.add_graph(sess.graph)

            summ = sess.run(summaries, feed_dict=data)
            writer.add_summary(summ, global_step=i)
            if train_accuracy >= 0.9:
                break
    finally:
        save_path = saver.save(sess, "data/71x40/model.ckpt")
