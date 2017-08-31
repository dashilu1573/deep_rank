#!/usr/bin/env python
# -*-coding: utf-8-*-

import os
import logging
import datetime
from sklearn import metrics
import tensorflow as tf
import numpy as np
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat
from tensorflow.python import debug as tf_debug
from models import *
from reader import *

# Define hyperparameters
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("input_file_format", "tfrecord", "Input file format")
flags.DEFINE_string("train_file", "./data/cancer_train.csv.tfrecords", "Train TFRecords files")
flags.DEFINE_string("validate_file", "./data/cancer_test.csv.tfrecords", "Validate TFRecords files")
flags.DEFINE_integer("feature_size", 9, "Number of feature size")
flags.DEFINE_integer("label_size", 2, "Number of label size")

flags.DEFINE_string("mode", "train", "Support train, export, inference, savedmodel")
flags.DEFINE_float("learning_rate", 0.001, "The learning rate")
flags.DEFINE_integer("epoch_number", 500, "Number of epochs to train")
flags.DEFINE_integer("batch_size", 200, "The batch size of training")
flags.DEFINE_integer("validate_batch_size", 100, "The batch size of validation")
flags.DEFINE_string("model", "deep", "Support deep, wide, wide_and_deep")
flags.DEFINE_string("model_network", "128 32 8", "The neural network of model")
flags.DEFINE_float("lr_decay_rate", 0.96, "Learning rate decay rate")
flags.DEFINE_string("optimizer", "rmsprop", "The optimizer to train")
flags.DEFINE_integer("steps_to_validate", 10, "Steps to validate and print state")

flags.DEFINE_string("checkpoint_path", "./checkpoint/", "The path of checkpoint")
flags.DEFINE_string("output_path", "./tensorboard/", "The path of tensorboard event files")
flags.DEFINE_string("model_path", "./train/", "The path of the saved model")

FEATURE_SIZE = FLAGS.feature_size
LABEL_SIZE = FLAGS.label_size
EPOCH_NUMBER = FLAGS.epoch_number
INPUT_FILE_FORMAT = FLAGS.input_file_format
if INPUT_FILE_FORMAT not in ["tfrecord", "csv"]:
    logging.error("Unknow input file format: {}".format(INPUT_FILE_FORMAT))
    exit(1)
OUTPUT_PATH = FLAGS.output_path
MODE = FLAGS.mode
CHECKPOINT_PATH = FLAGS.checkpoint_path
if not os.path.exists(CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH)
CHECKPOINT_FILE = CHECKPOINT_PATH + "/checkpoint.ckpt"

# 配置输出方式与日志级别
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='logger.log',
                    filemode='w')
# 定义一个StreamHandler，将INFO级别或更高的日志信息打印到标准错误，并将其添加到当前的日志处理对象#
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# Define the model
input_units = FEATURE_SIZE
output_units = LABEL_SIZE
model_network_hidden_units = [int(i) for i in FLAGS.model_network.split()]


def model(inputs):
    if FLAGS.model == "wide":
        return wide_model(inputs, input_units, output_units)
    elif FLAGS.model == "deep":
        return deep_model(inputs, input_units, output_units, model_network_hidden_units)
    elif FLAGS.model == "wide_n_deep":
        return wide_and_deep_model(inputs, input_units, output_units, model_network_hidden_units)
    else:
        logging.error("Unknown model, exit!")
        exit()


def get_optimizer(optimizer, learning_rate):
    # Opitmizer
    if optimizer == "sgd":
        return tf.train.GradientDescentOptimizer(learning_rate)
    elif optimizer == "adadelta":
        return tf.train.AdadeltaOptimizer(learning_rate, rho=0.95, epsilon=1e-08)
    elif optimizer == "adagrad":
        return tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value=0.1)
    elif optimizer == "adam":
        return tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)
    elif optimizer == "ftrl":
        return tf.train.FtrlOptimizer(learning_rate,
                                      learning_rate_power=-0.5,
                                      initial_accumulator_value=0.1,
                                      l1_regularization_strength=0.0,
                                      l2_regularization_strength=0.0)
    elif optimizer == "rmsprop":
        return tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10)
    else:
        logging.error("Unknow optimizer, exit!")
        exit()


def restore_session_from_checkpoint(sess, saver, checkpoint):
    if checkpoint:
        logging.info("Restore session from checkpoint: {}".format(checkpoint))
        saver.restore(sess, checkpoint)
        return True
    else:
        return False


def export_model(sess, model_path):
    logging.info("Export the model to {}".format(model_path))
    builder = tf.saved_model.builder.SavedModelBuilder(model_path)
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.TRAINING])
    builder.save()


def main():
    # Get hyperparameters

    # Read TFRecords files for training
    filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(FLAGS.train_file),
                                                    num_epochs=EPOCH_NUMBER)
    if INPUT_FILE_FORMAT == "tfrecord":
        label, features = read_and_decode_tfrecord(filename_queue, FEATURE_SIZE)
    elif INPUT_FILE_FORMAT == "csv":
        label, features = read_and_decode_csv(filename_queue)

    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * FLAGS.batch_size
    batch_labels, batch_features = tf.train.shuffle_batch(
        [label, features],
        batch_size=FLAGS.batch_size,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue)

    # Read TFRecords file for validatioin
    validate_filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(FLAGS.validate_file),
                                                             num_epochs=EPOCH_NUMBER)
    if INPUT_FILE_FORMAT == "tfrecord":
        validate_label, validate_features = read_and_decode_tfrecord(validate_filename_queue, FEATURE_SIZE)
    elif INPUT_FILE_FORMAT == "csv":
        validate_label, validate_features = read_and_decode_csv(validate_filename_queue)
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * FLAGS.validate_batch_size
    validate_batch_labels, validate_batch_features = tf.train.shuffle_batch(
        [validate_label, validate_features],
        batch_size=FLAGS.validate_batch_size,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue)

    # Loss
    logits = model(batch_features)
    batch_labels = tf.to_int64(batch_labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=batch_labels)
    loss = tf.reduce_mean(cross_entropy, name='loss')

    learning_rate = FLAGS.learning_rate
    optimizer = get_optimizer(FLAGS.optimizer, learning_rate)

    with tf.device("/cpu:0"):   # better than gpu
        global_step = tf.Variable(0, name='global_step', trainable=False)

    train_op = optimizer.minimize(loss, global_step=global_step)
    tf.get_variable_scope().reuse_variables()

    # Accuracy op for train data
    train_accuracy_logits = model(batch_features)
    train_softmax = tf.nn.softmax(train_accuracy_logits)
    train_correct_prediction = tf.equal(tf.argmax(train_softmax, 1), batch_labels)
    train_accuracy = tf.reduce_mean(tf.cast(train_correct_prediction, tf.float32))

    # Auc op for train data
    batch_labels = tf.cast(batch_labels, tf.int32)
    sparse_labels = tf.reshape(batch_labels, [-1, 1])
    derived_size = tf.shape(batch_labels)[0]
    indices = tf.reshape(tf.range(0, derived_size, 1), [-1, 1])
    concated = tf.concat(axis=1, values=[indices, sparse_labels])
    outshape = tf.stack([derived_size, LABEL_SIZE])
    new_batch_labels = tf.sparse_to_dense(concated, outshape, 1.0, 0.0)
    _, train_auc = tf.contrib.metrics.streaming_auc(train_softmax, new_batch_labels)

    # Accuracy op for validate data
    validate_accuracy_logits = model(validate_batch_features)
    validate_softmax = tf.nn.softmax(validate_accuracy_logits)
    validate_batch_labels = tf.to_int64(validate_batch_labels)
    validate_correct_prediction = tf.equal(tf.argmax(validate_softmax, 1), validate_batch_labels)
    validate_accuracy = tf.reduce_mean(tf.cast(validate_correct_prediction, tf.float32))

    # Auc op for validate data
    validate_batch_labels = tf.cast(validate_batch_labels, tf.int32)
    sparse_labels = tf.reshape(validate_batch_labels, [-1, 1])
    derived_size = tf.shape(validate_batch_labels)[0]
    indices = tf.reshape(tf.range(0, derived_size, 1), [-1, 1])
    concated = tf.concat(axis=1, values=[indices, sparse_labels])
    outshape = tf.stack([derived_size, LABEL_SIZE])
    new_validate_batch_labels = tf.sparse_to_dense(concated, outshape, 1.0, 0.0)
    _, validate_auc = tf.contrib.metrics.streaming_auc(validate_softmax, new_validate_batch_labels)

    # Inference op
    inference_features = tf.placeholder("float", [None, FEATURE_SIZE])
    inference_logits = model(inference_features)
    inference_softmax = tf.nn.softmax(inference_logits)  # 此处的softmax是否可以省略
    inference_op = tf.argmax(inference_softmax, 1)
    keys_placeholder = tf.placeholder(tf.int32, shape=[None, 1])
    keys = tf.identity(keys_placeholder)

    # Initialize saver and summary
    saver = tf.train.Saver()
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("train_accuracy", train_accuracy)
    tf.summary.scalar("train_auc", train_auc)
    tf.summary.scalar("validate_accuracy", validate_accuracy)
    tf.summary.scalar("validate_auc", validate_auc)
    summary_op = tf.summary.merge_all()

    # Create session to run 在一个会话中启动图
    init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(OUTPUT_PATH, sess.graph)
        sess.run(init_op)
        if "train" == MODE:
            # Restore session and start queue runner
            latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_PATH)
            restore_session_from_checkpoint(sess, saver, latest_checkpoint)

            # 创建一个协调器，管理线程
            coord = tf.train.Coordinator()
            # 启动队列
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            # tfdbg使用
            # https://github.com/tensorflow/tensorflow/issues/11017
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

            logging.info("开始训练!")
            start_time = datetime.datetime.now()
            try:
                while not coord.should_stop():
                    _, loss_value, step = sess.run([train_op, loss, global_step])
                    if step % FLAGS.steps_to_validate == 0:
                        train_accuracy_value, train_auc_value, validate_accuracy_value, auc_value, summary_value = \
                            sess.run([train_accuracy, train_auc, validate_accuracy, validate_auc, summary_op])
                        end_time = datetime.datetime.now()
                        logging.info(
                            "[{}] Step: {}, loss: {}, train_acc: {}, train_auc: {}, valid_acc: {}, valid_auc: {}".
                                format(end_time - start_time, step, loss_value, train_accuracy_value, train_auc_value,
                                       validate_accuracy_value, auc_value))
                        writer.add_summary(summary_value, step)
                        saver.save(sess, CHECKPOINT_FILE, global_step=step)
                        start_time = end_time
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
                # Export the model after training
                export_model(sess, FLAGS.model_path)
            finally:
                coord.request_stop()
                coord.join(threads)

            logging.info("训练结束!")

        elif MODE == "export":
            latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_PATH)
            if not restore_session_from_checkpoint(sess, saver, latest_checkpoint):
                logging.error("No checkpoint found, exit now")
                exit(1)

            # Export the model
            export_model(sess, FLAGS.model_path)

        elif MODE == "savedmodel":
            latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_PATH)
            if not restore_session_from_checkpoint(sess, saver, latest_checkpoint):
                logging.error("No checkpoint found, exit now")
                exit(1)

            logging.info(
                "Export the saved model to {}".format(FLAGS.saved_model_path))
            export_path_base = FLAGS.saved_model_path
            export_path = os.path.join(
                compat.as_bytes(export_path_base),
                compat.as_bytes(str(FLAGS.model_version)))

            model_signature = signature_def_utils.build_signature_def(
                inputs={
                    "keys": utils.build_tensor_info(keys_placeholder),
                    "features": utils.build_tensor_info(inference_features)
                },
                outputs={
                    "keys": utils.build_tensor_info(keys),
                    "softmax": utils.build_tensor_info(inference_softmax),
                    "prediction": utils.build_tensor_info(inference_op)
                },
                method_name=signature_constants.PREDICT_METHOD_NAME)

            try:
                builder = saved_model_builder.SavedModelBuilder(export_path)
                builder.add_meta_graph_and_variables(
                    sess,
                    [tag_constants.SERVING],
                    clear_devices=True,
                    signature_def_map={
                        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                            model_signature,
                    },
                    # legacy_init_op=legacy_init_op)
                    legacy_init_op=tf.group(
                        tf.initialize_all_tables(), name="legacy_init_op"))
                builder.save()
            except Exception as e:
                logging.error("Fail to export saved model, exception: {}".format(e))

        elif MODE == "inference":
            # Restore session and start queue runner
            latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_PATH)
            if not restore_session_from_checkpoint(sess, saver, latest_checkpoint):
                logging.error("No checkpoint found, exit now")
                exit(1)

            # Load inference test data(csv形式)
            inference_result_file_name = FLAGS.inference_result_file
            inference_test_file_name = FLAGS.inference_test_file
            inference_data = np.genfromtxt(inference_test_file_name, delimiter=",")
            inference_data_features = inference_data[:, 0:FEATURE_SIZE]
            inference_data_labels = inference_data[:, FEATURE_SIZE]

            # Run inference
            start_time = datetime.datetime.now()
            prediction, prediction_softmax = sess.run(
                [inference_op, inference_softmax],
                feed_dict={inference_features: inference_data_features})
            end_time = datetime.datetime.now()

            # Compute accuracy
            label_number = len(inference_data_labels)
            correct_label_number = 0
            for i in range(label_number):
                if inference_data_labels[i] == prediction[i]:
                    correct_label_number += 1
            accuracy = float(correct_label_number) / label_number

            # Compute auc
            y_true = np.array(inference_data_labels)
            y_score = prediction_softmax[:, 1]
            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            logging.info("[{}] Inference accuracy: {}, auc: {}".format(end_time - start_time, accuracy, auc))

            # Save result into the file
            np.savetxt(inference_result_file_name, prediction_softmax, delimiter=",")
            logging.info("Save result to file: {}".format(inference_result_file_name))


if __name__ == '__main__':
    main()
