import tensorflow as tf
from model import load_data, load_model

x, y, output, keep_prob = load_model()

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=y))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

data_x, data_y = load_data(x, y)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess.run(init)

    correct_prediction = tf.equal(tf.argmax(output, 1), y)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.histogram("normal/accuracy", accuracy)
    try:
        for i in range(9999):
            sess.run(train_step, feed_dict={x: data_x,
                                            y: data_y,
                                            keep_prob: 0.75})
            if i % 100 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={x: data_x,
                                                               y: data_y,
                                                               keep_prob: 1.})
                print('Step {:5d}: training accuracy {:g}'.format(i, train_accuracy))
                if train_accuracy >= 0.95:
                    break
    finally:
        save_path = saver.save(sess, "data/71x40/model.ckpt")
