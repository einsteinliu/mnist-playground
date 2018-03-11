import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data
def init_random_weights(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def init_bias_weights(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

with tf.Session() as sess:
  with tf.device("/cpu:0"):
        #x and y_ can be any thing, they are just place holders
        x = tf.placeholder(tf.float32, shape = [None,784])
        y_ = tf.placeholder(tf.float32,shape = [None,10])
        x_image = tf.reshape(x, [-1,28,28,1])

        #first layer
        #window w=5, h=5, image depth=1, output features = 32
        kernel1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], dtype=tf.float32,stddev=1e-1), name='weights')
        conv1 = tf.nn.conv2d(x_image,kernel1,[1,1,1,1],padding='SAME')
        bias1 = tf.Variable(tf.constant(0.1,shape=[32], dtype=tf.float32),trainable=True, name='biases')
        out1 = tf.nn.bias_add(conv1,bias1)
        conv1_act = tf.nn.relu(out1)
        # 1:from one channel, we don't want to take the maximum over multiple examples, or over multiples channels.
        # 2x2 window size to get 1 maximum
        pool1 = tf.nn.max_pool(conv1_act,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

        #The second layer
        kernel2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], dtype=tf.float32,
                                                stddev=1e-1), name='weights')
        conv2 = tf.nn.conv2d(pool1,kernel2,[1,1,1,1],padding='SAME')
        bias2 = tf.Variable(tf.constant(0.1,shape=[64], dtype=tf.float32),
                            trainable=True, name='biases')
        out2 = tf.nn.bias_add(conv2,bias2)
        conv2_act = tf.nn.relu(out2)
        # 1:from one channel, we don't want to take the maximum over multiple examples, or over multiples channels.
        # 2x2 window size to get 1 maximum
        pool2 = tf.nn.max_pool(conv2_act,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        
        #fc
        shape = int(np.prod(pool2.get_shape()[1:]))
        fc1w = tf.Variable(tf.truncated_normal([shape, 1024],dtype=tf.float32,stddev=1e-1), name='weights')
        fc1b = tf.Variable(tf.constant(0.1, shape=[1024], dtype=tf.float32),trainable=True, name='biases')
        pool2_flat = tf.reshape(pool2, [-1, shape])
        fc1l = tf.nn.bias_add(tf.matmul(pool2_flat, fc1w), fc1b)
        fc1 = tf.nn.relu(fc1l)

        keep_p = tf.placeholder(tf.float32)
        fc1_dropout = tf.nn.dropout(fc1,keep_p)

        fc2w = tf.Variable(tf.truncated_normal([1024, 10],dtype=tf.float32,stddev=1e-1), name='weights')
        fc2b = tf.Variable(tf.constant(0.1, shape=[10], dtype=tf.float32),trainable=True, name='biases')
        fc2l = tf.nn.bias_add(tf.matmul(fc1_dropout, fc2w), fc2b)
        y = tf.nn.relu(fc2l)

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))
        train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

        # tf.device('/gpu:0')
        # sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
        sess.run(tf.global_variables_initializer())
        results = []
        start = time.time()
        for step in range(20000):
            batch = mnist.train.next_batch(50)
            train_step.run(feed_dict = {x:batch[0],y_:batch[1],keep_p:0.5})
            #train_step.run(feed_dict = {x:batch[0],y_:batch[1]})
            if step%100==0:
                correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
                correct_ratio = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
                #result = correct_ratio.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels})
                result = correct_ratio.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_p:1.0})
                print(step,str(result))
                results.append(result)
                f = open('log.txt','w')
                for accu in results:
                    f.write(str(accu)+'\n')
                f.close();
        end = time.time()
        print(str(end-start)+' s')
        plt.plot(results)
        plt.show()
