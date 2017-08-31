#!/usr/bin/env python
# -*-coding: utf-8-*-

import tensorflow as tf

BUCKET_SIZE = 100


#  读取tfrecord文件
def read_and_decode_tfrecord(filename_queue, feature_size):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            "label": tf.FixedLenFeature([], tf.float32),
            "features": tf.FixedLenFeature([feature_size], tf.float32),
        })
    label = features["label"]
    features = features["features"]
    return label, features


# 读取csv文件
def read_and_decode_csv(filename_queue):
    # TODO: Not generic for all datasets
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)

    # Default values, in case of empty columns. Also specifies the type of the decoded result.
    record_defaults = [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]
    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = tf.decode_csv(value, record_defaults=record_defaults)
    label = col10
    features = tf.stack([col1, col2, col3, col4, col5, col6, col7, col8, col9])
    return label, features