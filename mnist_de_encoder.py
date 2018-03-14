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
    groundtruth = tf.reshape(labels,[-1,28,28,1])
    
    arg_scope_conv2d = tf.contrib.framework.arg_scope([layers.conv2d],
                                        kernel_size=[5,5],
                                        weights_initializer=tf.initializers.truncated_normal(stddev=0.1),                                        
                                        activation_fn = tf.nn.leaky_relu,
                                        biases_initializer=tf.initializers.constant(0.1),
                                        normalizer_fn=layers.batch_norm)

    arg_scope_deconv2d = tf.contrib.framework.arg_scope([layers.conv2d_transpose],
                                        kernel_size=[2,2],
                                        stride = 2,
                                        weights_initializer=tf.initializers.truncated_normal(stddev=0.1),                                        
                                        activation_fn = tf.nn.leaky_relu,
                                        biases_initializer=tf.initializers.constant(0.1),
                                        normalizer_fn=layers.batch_norm)

    arg_scope_full_connected = tf.contrib.framework.arg_scope([layers.fully_connected],
                                                              activation_fn=tf.nn.leaky_relu,
                                                              biases_initializer = tf.initializers.constant(0.1))
    
    with arg_scope_conv2d:
        with arg_scope_full_connected:
            with arg_scope_deconv2d:
                conv1 = tf.contrib.layers.conv2d(input_layer,num_outputs=32)
                pool1 = tf.contrib.layers.max_pool2d(conv1,kernel_size=[2,2],stride=2)
                conv2 = tf.contrib.layers.conv2d(pool1,num_outputs=64)
                pool2 = tf.contrib.layers.max_pool2d(conv2,kernel_size=[2,2],stride=2)

                shp = pool2.get_shape()
                flat1 = tf.contrib.layers.flatten(pool2)
                fc1 = tf.contrib.layers.fully_connected(flat1,num_outputs=128)
                fc2 = tf.contrib.layers.fully_connected(fc1,num_outputs=7*7*64)
                deflat2 = tf.reshape(fc2,[-1,7,7,64])

                conv21 = tf.contrib.layers.conv2d(deflat2,num_outputs=64)
                deconv21 = tf.contrib.layers.conv2d_transpose(conv21,num_outputs=32)
                conv22 = tf.contrib.layers.conv2d(deconv21,num_outputs=32)
                reconstruct_logits = tf.contrib.layers.conv2d_transpose(conv22,num_outputs=1,activation_fn=None)                
                
                reconstruct_probs = tf.nn.sigmoid(reconstruct_logits,name = "softmax")
                reconstruct_result = tf.cast(tf.greater(reconstruct_probs,0.5),dtype=tf.float32)
                
                #shape_tmp = reconstruct_result.get_shape()
                tf.summary.image("Diff",reconstruct_result-groundtruth)      
                tf.summary.image("result",reconstruct_probs)      
                tf.summary.image("origin",groundtruth)
                reconstruction = {
                    "probabilities":reconstruct_probs
                    }

                if mode == tf.estimator.ModeKeys.PREDICT:
                    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
                
                loss = tf.losses.absolute_difference(labels=input_layer,predictions=reconstruct_probs)

                if mode == tf.estimator.ModeKeys.TRAIN:
                    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
                    training_op = optimizer.minimize(
                        loss = loss,
                        global_step = tf.train.get_global_step())
                    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=training_op)

def main(args):
    mnist = input_data.read_data_sets('MNIST_data')
    train_image = mnist.train.images    
    train_labels = np.array(mnist.train.labels,dtype=np.int32)

    #extract 10 pattern image
    pattern_image = []
    for i in range(0,10):
        j = 0
        while(train_labels[j]!=i):
            j = j + 1
        pattern_image.append(train_image[j])            
    train_image_labels = []

    #construct ground truth
    for i in range(0,train_labels.shape[0]):
        train_image_labels.append(pattern_image[train_labels[i]])
    train_image_labels = np.asarray(train_image_labels)    

    estimator = tf.estimator.Estimator(
        model_fn = lenet_with_scope,
        model_dir = "D:/Study/DeepLearning/MNIST_Playground/mnist-playground/ckp")

    tensors_to_log = {"probabilities": "softmax"}
    logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)
    
    train_func = tf.estimator.inputs.numpy_input_fn(
        x = {"x":train_image},
        y = train_image_labels,
        batch_size = 50,
        num_epochs=None,
        shuffle=True)

    estimator.train(
        input_fn = train_func,
        steps = 200000,
        hooks = [logging_hook])

if __name__ == "__main__":
  tf.app.run()
