import tensorflow as tf
import numpy as np
from numpy import array
from PIL import Image
from flask import Flask, request, jsonify
from io import BytesIO
from model import load_loopable_model, map_conv, map_func

app = Flask(__name__)

x, y, output, keep_prob = load_loopable_model()

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
