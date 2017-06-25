import tensorflow as tf
import numpy as np
from numpy import array
from PIL import Image
from flask import Flask, request, jsonify
from io import BytesIO

app = Flask(__name__)

n_nodes_hl1 = 75
n_nodes_hl2 = 75
n_nodes_hl3 = 75
n_nodes_hl4 = 75

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


def map_func(x):
    return np.float32(x[0] / 255)


def map_conv(x):
    return np.float32(x)


sess = tf.Session()
init = tf.global_variables_initializer()
saver = tf.train.Saver()
saver.restore(sess, "data/71x40/model.ckpt")


@app.route('/', methods=['POST'])
def parse_request():
    img = Image.open(BytesIO(request.files["file"].read()))
    data = list(map(map_func, array(img.getdata(), np.uint8)))
    var = sess.run(tf.argmax(output, 1), {x: [data]})
    return jsonify(str(var[0]))


@app.route('/json', methods=['POST'])
def parse_json_request():
    data = list(map(map_conv, request.get_json()))
    var = sess.run(tf.argmax(output, 1), {x: [data]})
    return jsonify(str(var[0]))


app.run()
