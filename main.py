import tensorflow as tf
from model import load_data, load_model

x, y, output = load_model()

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=y))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

data = load_data(x, y)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess.run(init)

    correct_prediction = tf.equal(tf.argmax(output, 1), y)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.histogram("normal/accuracy", accuracy)
    try:
        for i in range(1000):
            train_accuracy = sess.run(accuracy, feed_dict=data)
            sess.run(train_step, feed_dict=data)
            print('Step {:5d}: training accuracy {:g}'.format(i, train_accuracy))
            if train_accuracy >= 1:
                break
    finally:
        save_path = saver.save(sess, "data/71x40/model.ckpt")
