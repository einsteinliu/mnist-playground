from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow.contrib.layers as layers
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import framework
from tensorflow.contrib.framework.python.ops.arg_scope import add_arg_scope

tf.logging.set_verbosity(tf.logging.INFO)

def lenet_with_scope(features,labels,mode):
    input_layer = tf.reshape(features["x"],[-1,28,28,1])
    arg_scope_conv2d = tf.contrib.framework.arg_scope([layers.conv2d],
                                        kernel_size=[5,5],
                                        weights_initializer=tf.initializers.truncated_normal(stddev=0.1),                                        
                                        activation_fn = tf.nn.relu,
                                        biases_initializer=tf.initializers.constant(0.1))

    arg_scope_full_connected = tf.contrib.framework.arg_scope([layers.fully_connected],
                                                              activation_fn=tf.nn.relu,
                                                              biases_initializer = tf.initializers.constant(0.1))
    with arg_scope_conv2d:
        with arg_scope_full_connected:
            conv1 = tf.contrib.layers.conv2d(input_layer,num_outputs=32)
            pool1 = tf.contrib.layers.max_pool2d(conv1,kernel_size=[2,2],stride=2)
            conv2 = tf.contrib.layers.conv2d(pool1,num_outputs=64)
            pool2 = tf.contrib.layers.max_pool2d(conv2,kernel_size=[2,2],stride=2)

            flat1 = tf.contrib.layers.flatten(pool2)         
            fc1 = tf.contrib.layers.fully_connected(inputs=flat1,num_outputs=1024,scope="fc1")
            dropout1 = tf.nn.dropout(x=fc1,keep_prob=0.4)
            logits = tf.contrib.layers.fully_connected(inputs=dropout1,num_outputs=10,scope="logits")

            prediction = {
                "class":tf.argmax(input=logits, axis=1),
                "probabilities":tf.nn.softmax(logits,name = "softmax")
                }

            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

            #loss = tf.losses.sparse_softmax_cross_entropy(
            #    labels = labels,
            #    logits = logits)
   
            one_hot_labels = tf.one_hot(labels,10)
            loss = tf.losses.softmax_cross_entropy(one_hot_labels,logits)   
            pers = tf.reduce_sum(tf.pow(one_hot_labels-logits,2))
            tf.summary.scalar("per",pers)

            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
                training_op = optimizer.minimize(
                    loss = loss,
                    global_step = tf.train.get_global_step())
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=training_op)

def lenet(features,labels,mode):  
    
   input_layer = tf.reshape(features["x"],[-1,28,28,1])
   #pattern_image = tf.reshape(features["pattern"],[-1,28,28,1])
 
   conv1 = tf.layers.conv2d(
       inputs = input_layer,
       filters=32,
       kernel_size=[5,5],
       kernel_initializer = tf.truncated_normal_initializer(stddev=0.1),
       padding = "same",
       activation = tf.nn.relu,
       use_bias=True,
       bias_initializer = tf.constant_initializer(0.1),
       name="Conv1")
   pool1 = tf.layers.max_pooling2d(
       inputs=conv1,
       pool_size=[2,2],
       strides=2)
   
   conv2 = tf.layers.conv2d(
       inputs = pool1,
       filters=64,
       kernel_size=[5,5],
       kernel_initializer = tf.truncated_normal_initializer(stddev=0.1),
       padding = "same",
       activation = tf.nn.relu,
       use_bias=True,
       bias_initializer = tf.constant_initializer(0.1),
       name="Conv2")
   pool2 = tf.layers.max_pooling2d(
       inputs=conv2,
       pool_size=[2,2],
       strides=2)

   flat1 = tf.layers.flatten(pool2)   

   fc1 = tf.layers.dense(
       inputs=flat1,
       units=1024,            
       activation=tf.nn.relu,
       use_bias = True,
       bias_initializer = tf.constant_initializer(0.1),
       name="fc1")
   dropout1 = tf.layers.dropout(
       inputs=fc1,
       rate=0.4,
       training=mode == tf.estimator.ModeKeys.TRAIN)

   logits = tf.layers.dense(
       inputs=dropout1,
       units=10,     
       activation=tf.nn.relu,
       use_bias = True,
       bias_initializer = tf.constant_initializer(0.1),
       name="logits")

   prediction = {
       "class":tf.argmax(input=logits, axis=1),
       "probabilities":tf.nn.softmax(logits,name = "softmax")
       }

   if mode == tf.estimator.ModeKeys.PREDICT:
       return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

   #loss = tf.losses.sparse_softmax_cross_entropy(
   #    labels = labels,
   #    logits = logits)
   
   one_hot_labels = tf.one_hot(labels,10)
   loss = tf.losses.softmax_cross_entropy(one_hot_labels,logits)   
   pers = tf.reduce_sum(tf.pow(one_hot_labels-logits,2))
   tf.summary.scalar("per",pers)

   if mode == tf.estimator.ModeKeys.TRAIN:
       optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
       training_op = optimizer.minimize(
           loss = loss,
           global_step = tf.train.get_global_step())
       return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=training_op)

def main(args):
    mnist = input_data.read_data_sets('MNIST_data')
    train_image = mnist.train.images    
    train_labels = np.array(mnist.train.labels,dtype=np.int32)

    pattern_image = []
    for i in range(0,10):
        j = 0
        while(train_labels[j]!=i):
            j = j + 1
        pattern_image.append(train_image[j])
    pattern_image = np.asarray(pattern_image)   

    estimator = tf.estimator.Estimator(
        model_fn = lenet_with_scope,        
        model_dir = "D:/Study/DeepLearning/TensorFlow/ckp")

    tensors_to_log = {"probabilities": "softmax"}
    logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)
    
    train_func = tf.estimator.inputs.numpy_input_fn(
        x = {"x":train_image},
        y = train_labels,
        batch_size = 50,
        num_epochs=None,
        shuffle=True)

    estimator.train(
        input_fn = train_func,
        steps = 200000,
        hooks = [logging_hook])

if __name__ == "__main__":
  tf.app.run()