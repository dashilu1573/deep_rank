#!/usr/bin/env python
# -*-coding: utf-8-*-

import tensorflow as tf


def full_connect(inputs, weights_shape, biases_shape):
    # with tf.device('/cpu:0'):  # for better performance ???????

    # tf.Variable和tf.get_variable的区别
    weights = tf.get_variable("weights", weights_shape, initializer=tf.random_normal_initializer())
    biases = tf.get_variable("biases", biases_shape, initializer=tf.random_normal_initializer())
    layer = tf.matmul(inputs, weights) + biases

    return layer


def full_connect_relu(inputs, weights_shape, biases_shape):
    return tf.nn.relu(full_connect(inputs, weights_shape, biases_shape))


def deep_model(inputs, input_units, output_units, model_network_hidden_units):
    with tf.variable_scope("input"):
        layer = full_connect_relu(inputs,
                                  [input_units, model_network_hidden_units[0]],
                                  [model_network_hidden_units[0]])

    for i in range(len(model_network_hidden_units) - 1):
        with tf.variable_scope("layer{}".format(i)):
            layer = full_connect_relu(layer,
                                      [model_network_hidden_units[i], model_network_hidden_units[i + 1]],
                                      [model_network_hidden_units[i + 1]])

    with tf.variable_scope("output"):
        layer = full_connect(layer,
                             [model_network_hidden_units[-1], output_units],
                             [output_units])
    return layer


def wide_model(inputs, input_units, output_units):
    with tf.variable_scope("logistic regression"):
        layer = full_connect(inputs, [input_units, output_units], [output_units])
    return layer


def wide_and_deep_model(inputs, input_units, output_units, model_network_hidden_units):
    return wide_model(inputs, input_units, output_units) + \
           deep_model(inputs, input_units, output_units, model_network_hidden_units)


def wide_and_deep_cnn_model(inputs, input_units, output_units, model_network_hidden_units):
    return wide_model(inputs, input_units, output_units) + \
           deep_model(inputs, input_units, output_units, model_network_hidden_units) + \
           cnn_model(inputs, output_units)


def cnn_model(inputs, output_units):
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        inputs = tf.reshape(inputs, [-1, 64, 64, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(inputs, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 16x16x64 feature maps -- maps this to 128 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([16 * 16 * 64, output_units])
        b_fc1 = bias_variable([output_units])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 16 * 16 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    return h_fc1_drop


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
