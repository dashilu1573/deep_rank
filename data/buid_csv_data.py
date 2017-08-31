#!/usr/bin/env python
# -*-coding: utf-8-*-

import tensorflow as tf
import os
import sys
import multiprocessing

FEATURE_SIZE = 9


def convert_tfrecords(input_filename, output_filename):
    print("Start to convert {} to {}".format(input_filename, output_filename))
    writer = tf.python_io.TFRecordWriter(output_filename)

    for line in open(input_filename, "r"):
        data = line.split(",")
        label = float(data[FEATURE_SIZE])
        features = [float(i) for i in data[:FEATURE_SIZE]]

        example = tf.train.Example(features=tf.train.Features(
            feature={
                "label": tf.train.Feature(float_list=tf.train.FloatList(value=[label])),
                "features": tf.train.Feature(float_list=tf.train.FloatList(value=features)),
            }
        ))

        writer.write(example.SerializeToString())
    writer.close()
    print("Successfully convet {} to {}".format(input_filename, output_filename))


def main():
    pool = multiprocessing.Pool(processes=10)
    workers = []

    data_path = sys.argv[1]
    for filename in os.listdir(data_path):
        if filename.endswith(".csv"):
            workers.append(pool.apply_async(convert_tfrecords, (filename, filename + ".tfrecords")))
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
