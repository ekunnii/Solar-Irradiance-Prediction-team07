from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras import datasets, layers, models, optimizers, metrics
from tensorflow import keras
from tensorflow.python.ops import summary_ops_v2

import argparse
import datetime
import typing
import json
import os
import tensorflow as tf
import logging
import pdb
import time
import pandas as pd

import matplotlib.pyplot as plt
import pdb
import time
import logging
import pandas as pd
import numpy as np

from models.model_factory import ModelFactory
from dataloader.dataset import TrainingDataSet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

"""
train loop credit to https://github.com/dragen1860/TensorFlow-2.x-Tutorials/blob/master/01-TF2.0-Overview/conv_train.py
"""

tf.keras.backend.set_floatx('float64')

def extract_data_frame_path(train_config: json):
    """
    It checks if we have a file
    """
    if os.path.isfile(train_config["dataframe_path"]):
        return train_config["dataframe_path"]
    else:
        path = os.getcwd() + train_config["relative_dataframe_path"]
        if os.path.isfile(path):
            return path

    assert True, f"Unable to find training data frame file from: {path} or from {train_config.dataframe_path}"


def extract_station_offsets(train_config: json):
    stations = train_config["stations"]
    target_time_offsets = train_config["target_time_offsets"]
    return stations, target_time_offsets

def plot_loss(train_losses, eval_losses):
    x = range(len(train_losses))
    plt.figure(figsize=[12, 8])
    plt.grid(color='grey', linestyle='-', linewidth=0.5)
    plt.title("Solar train and eval losses")
    plt.plot(x, train_losses, label='train losses')
    plt.plot(x, eval_losses, label='eval losses')
    plt.legend(['train losses', 'eval losses'])
    plt.xlabel('check points per 1000 steps')
    plt.ylabel('losses')
    plt.show()


def apply_clean(dirname):
    """delete directory
    
    Arguments:
        dirname {[type]} -- [description]
    """    
    if tf.io.gfile.exists(dirname):
        print('Removing existing dir: {}'.format(dirname))
        tf.io.gfile.rmtree(dirname)

def solar_datasets(datasets):
    """
    train and eaval split

    """    
    print("*******Create training dataset********")
    if not args.dont_use_cache:
        train_ds = TrainingDataSet(data_frame_path, stations, train_json, user_config=user_config_json, scratch_dir=args.scratch_dir) \
            .prefetch(tf.data.experimental.AUTOTUNE) \
            .batch(batch_size) \
            .cache(cache_dir + "/tf_learn_cache") \
            .shuffle(buffer_size)
    else:
        train_ds = TrainingDataSet(data_frame_path, stations, train_json, user_config=user_config_json, scratch_dir=args.scratch_dir) \
            .prefetch(tf.data.experimental.AUTOTUNE) \
            .batch(batch_size)

    eval_ds = train_ds

    return train_ds, eval_ds

def train_step(model, optimizer, meta_data, images, labels):

    # Record the operations used to compute the loss, so that the gradient
    # of the loss with respect to the variables can be computed.
    with tf.GradientTape() as tape:
        logits = model(meta_data, images, training=True)
        loss = compute_loss(labels, logits)
        # compute_accuracy(labels, logits)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss


def train(model, optimizer, dataset, log_freq=50):
    """
    Trains model on `dataset` using `optimizer`.
    """
    # Metrics are stateful. They accumulate values and return a cumulative
    # result when you call .result(). Clear accumulated values with .reset_states()
    avg_loss = metrics.Mean('loss', dtype=tf.float32)

    # Datasets can be iterated over like any other Python iterable.
    for (meta_data, images, labels) in dataset:
        loss = train_step(model, optimizer, meta_data, images, labels)
        avg_loss(loss)

        if tf.equal(optimizer.iterations % log_freq, 0):
            print("first sample from batch", images[0], labels[0])
            # summary_ops_v2.scalar('loss', avg_loss.result(), step=optimizer.iterations)
            # summary_ops_v2.scalar('accuracy', compute_accuracy.result(), step=optimizer.iterations)
            print('step:', int(optimizer.iterations),
                  'loss:', avg_loss.result().numpy(),
                  'RMSE:', np.sqrt(avg_loss.result().numpy()))
            avg_loss.reset_states()
            # compute_accuracy.reset_states()


def test(model, dataset, step_num):
    """
    Perform an evaluation of `model` on the examples from `dataset`.
    """
    avg_loss = metrics.Mean('loss', dtype=tf.float32)

    for (meta_data, images, labels) in dataset:
        logits = model(metas_data, images, training=False)
        avg_loss(compute_loss(labels, logits))
        # compute_accuracy(labels, logits)

    print('Model test set loss: {:0.4f} RMSE: {:0.2f}%'.format(
        avg_loss.result(), np.sqrt(avg_loss.result().numpy())))

    print('loss:', avg_loss.result(), 'RMSE:',
          np.sqrt(avg_loss.result().numpy()))
    # summary_ops_v2.scalar('loss', avg_loss.result(), step=step_num)
    # summary_ops_v2.scalar('accuracy', compute_accuracy.result(), step=step_num)

if __name__ == "__main__":
    print("Entering training python script.")

    # Arguments passed to training script
    parser = argparse.ArgumentParser()
    parser.add_argument("train_config", type=str,
                        help="Path of the training config file. This file contains ")
    parser.add_argument("-n", "--num_epochs", type=int, default=100,
                        help="Number of epochs we want the model to train")
    parser.add_argument("-m", "--model_name", type=str, default="DummyModel",
                        help="To train a specefic model, you can specify the name of your model. The ModelFactory will return the right model to train.\
                            The model name should be the modle class name. Example: 'DummyModel'. ")
    parser.add_argument("-u", "--user_config", type=str, default="",
                        help="Path to the JSON config file used to store user model/dataloader parameters")
    parser.add_argument("-s", "--scratch_dir", type=str, default=None,
                        help="Important for performance on the cluster!! If you want the files to be read fast, please set this variable.")
    parser.add_argument("--model_dir", type=str, default="./models",
                        help="Directory to save the checkpoints")
    parser.add_argument("--training", type=bool, default=True,
                        help="Enable training or not")
    parser.add_argument("--use_cache", type=bool, default=True,
                        help="Enable dataloader cache or not")
    parser.add_argument("--delete_checkpoints", type=bool, default=False,
                        help="Delete previous checkpoints or not, by default is False")
    parser.add_argument("--load_checkpoints", type=bool, default=True,
                        help="load previous checkpoints or not, by default is True")

    args = parser.parse_args()

    print("Starting Training!") 

    # Load configs
    assert os.path.isfile(
        args.train_config), f"Invalid training configuration file: {args.train_config}"
    with open(args.train_config, "r") as tc:
        train_json = json.load(tc)

    user_config_json = None
    if args.user_config:
        with open(args.user_config, "r") as uc:
            user_config_json = json.load(uc)

    cache_dir = args.scratch_dir or os.getcwd()
    batch_size = train_json.get("batch_size") or 32
    buffer_size = train_json.get("buffer_size") or 1000
    data_frame_path = extract_data_frame_path(train_json)
    stations, target_time_offsets = extract_station_offsets(train_json)

    # Init models, dataset and other vars for the training loop
    print("*******Create Model********")
    model_factory = ModelFactory(
        stations, target_time_offsets, args.user_config)
    model = model_factory.build(args.model_name)

    

    print("*******Create training dataset********")
    if args.use_cache:
        train_ds = TrainingDataSet(data_frame_path, stations, train_json, user_config=user_config_json, scratch_dir=args.scratch_dir) \
            .prefetch(tf.data.experimental.AUTOTUNE) \
            .batch(batch_size) \
            .cache(cache_dir + "/tf_learn_cache") \
            .shuffle(buffer_size)
    else:
        train_ds = TrainingDataSet(data_frame_path, stations, train_json, user_config=user_config_json, scratch_dir=args.scratch_dir) \
            .prefetch(tf.data.experimental.AUTOTUNE) \
            .batch(batch_size) ##\
            ##.shuffle(buffer_size)

    train_loss_results = []
    train_accuracy_results = []
    is_training = args.training
    compute_loss = tf.keras.losses.MSE
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.00000001)
    optimizer = optimizers.SGD(learning_rate=0.01, momentum=0.5)
    # compute_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()


    logging.basicConfig(filename='result.log',level=logging.DEBUG)

    # Where to save checkpoints, tensorboard summaries, etc.
    checkpoint_dir = os.path.join(args.model_dir, 'checkpoints')
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

    # clear previous checkpoints for debug purpose
    if args.delete_checkpoints:
        apply_clean(checkpoint_dir)

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

    # Restore variables on creation if a checkpoint exists.
    if args.load_checkpoints:
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    for i in range(args.num_epochs):
        start = time.time()
        #   with train_summary_writer.as_default():
        train(model, optimizer, train_ds, log_freq=5)
        end = time.time()
        print('Train time for epoch #{} ({} total steps): {}'.format(
            i + 1, int(optimizer.iterations), end - start))
        
        # with test_summary_writer.as_default():
        #     test(model, test_ds, optimizer.iterations)

        checkpoint.save(checkpoint_prefix)
        print('saved checkpoint.')

    # export_path = os.path.join(MODEL_DIR, 'export')
    # tf.saved_model.save(model, export_path)
    # print('saved SavedModel for exporting.')


    # # main loop
    # for epoch in range(args.num_epochs):
    #     datafetch_time = time.perf_counter()
    #     epoch_loss_avg = tf.keras.metrics.Mean()
    #     start_time = time.perf_counter()

    #     print("*******EPOCH %d start********" % (epoch+1))

    #     for iter_idx, (metas, images, targets) in enumerate(dataset):

    #         with tf.GradientTape() as tape:
    #             images = tf.keras.utils.normalize(images,axis=-1)

    #             y_ = model(metas, images, training=True)
    #             loss_value = loss_fct(y_true=targets, y_pred=y_)
    #             if iter_idx % 9 == 0:
    #                 # print("********predicted value")
    #                 # print(y_)
    #                 print(f"Batch loss: {loss_value}")
                
    #             print(f"Batch loss {iter_idx}: {np.mean(loss_value)}")
    #             #print(f"Batch loss: {loss_value}")

    #         grads = tape.gradient(loss_value, model.trainable_variables)
    #         optimizer.apply_gradients(zip(grads, model.trainable_variables))

    #         # Track progress
    #         epoch_loss_avg(loss_value)  # Add current batch loss
    #         if iter_idx % 9 == 0:
    #             print(
    #                 f"Data Fetch time: {time.perf_counter() - datafetch_time}, for batch size: {metas.shape[0]}")
    #             datafetch_time = time.perf_counter()

    #         if iter_idx % 9 == 0:
    #             print("epoch : %d , iter: %d,  epoch loss: %s" %
    #                   (epoch + 1, iter_idx + 1, loss_value))

    #     # End epoch
    #     train_loss_results.append(epoch_loss_avg.result())
    #     logging.debug(train_loss_results)
    #     print(f"Epoch result: {epoch_loss_avg.result()}")
    #     print(f"Elapsed time for epoch: {time.perf_counter() - start_time}")
