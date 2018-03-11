import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_input = 28
n_hidden_state = 128
n_time_steps = 28
n_classes = 10

learning_rate = 0.001
training_iters = 100000

w = tf.Variable(tf.random_normal([n_hidden_state, n_classes]))
b = tf.Variable(tf.random_normal([n_classes]))

x = tf.placeholder(tf.float32,[None,n_time_steps,n_input])
y = tf.placeholder(tf.float32,[None,n_classes])

x_ = tf.unstack(x,n_time_steps,1)
lstm_cell = rnn.rnn_cell.BasicLSTMCell(n_hidden_state,forget_bias=1.0)
outputs, states = rnn.rnn.rnn(lstm_cell, x_,initial_state=None,dtype=tf.float32)

y_ = tf.add(tf.matmul(outputs[-1],w),b)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_,1), tf.argmax(y,1)), tf.float32))

init = tf.global_variables_initializer()

results = []
with tf.Session() as sess:
    sess.run(init)
    for step in range(20000):
        batch_x,batch_y = mnist.train.next_batch(50)
        batch_x = batch_x.reshape(50,n_time_steps,n_input)
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
        if step % 50 == 0:
            currResult = sess.run(accuracy, feed_dict={x: mnist.test.images.reshape(mnist.test.images.shape[0],n_time_steps,n_input), y: mnist.test.labels})
            results.append(currResult)
            print(step,currResult)
            f = open('log.txt','w')
            for accu in results:
                f.write(str(accu)+'\n')
            f.close();
            # # Calculate batch accuracy
            # acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # # Calculate batch loss
            # loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            # print("Iter " + str(step) + ", Minibatch Loss= " + \
            #       "{:.6f}".format(loss) + ", Training Accuracy= " + \
            #       "{:.5f}".format(acc))        
    print("Optimization Finished!")


