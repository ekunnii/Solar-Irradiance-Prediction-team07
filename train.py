from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras import datasets, layers, models, optimizers, metrics
from tensorflow.keras.applications.resnet50 import preprocess_input  as preprocess_input_resnet50
from tensorflow import keras
from tensorflow.python.ops import summary_ops_v2

import argparse
import datetime
import typing
import json
import os
import tensorflow as tf
import pdb
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import time
import numpy as np

from models.model_factory import ModelFactory
from dataloader.dataset import TrainingDataSet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

"""
train loop credit to https://github.com/dragen1860/TensorFlow-2.x-Tutorials/blob/master/01-TF2.0-Overview/conv_train.py
"""

#tf.keras.backend.set_floatx('float64') #commented because it crashes resnet

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

def apply_clean(dirname):
    """delete directory

    Arguments:
        dirname {[type]} -- [description]
    """
    if tf.io.gfile.exists(dirname):
        print('Removing existing dir: {}'.format(dirname))
        tf.io.gfile.rmtree(dirname)

def solar_datasets():
    """
    train and valid split

    """
    print("*******Create training dataset********")
    if args.use_cache:
        train_ds = TrainingDataSet(data_frame_path, stations, train_json, user_config=user_config_json,
                                   scratch_dir=args.scratch_dir) \
            .prefetch(tf.data.experimental.AUTOTUNE) \
            .batch(batch_size) \
            .cache(cache_dir + "/tf_learn_cache") \
            .shuffle(buffer_size)

        valid_ds = TrainingDataSet(data_frame_path, stations, train_json, user_config=user_config_json,
                                 train=False, scratch_dir=args.scratch_dir) \
            .prefetch(tf.data.experimental.AUTOTUNE) \
            .batch(batch_size) \
            .cache(cache_dir + "/tf_learn_cache") \
            .shuffle(buffer_size)
    else:
        train_ds = TrainingDataSet(data_frame_path, stations, train_json, user_config=user_config_json,
                                   scratch_dir=args.scratch_dir) \
            .prefetch(tf.data.experimental.AUTOTUNE) \
            .batch(batch_size) \
            .shuffle(buffer_size)

        valid_ds = TrainingDataSet(data_frame_path, stations, train_json, user_config=user_config_json,
                                   train=False, scratch_dir=args.scratch_dir) \
            .prefetch(tf.data.experimental.AUTOTUNE) \
            .batch(batch_size) \
            .shuffle(buffer_size)

    return train_ds, valid_ds

def preprocess(images, meta_data):
    # if pretrained, must use the same preprocess as when the model was trained, here preprocess of resnet
    if 'pretrained_resnet' == args.model_name:
        # images = preprocess_input_resnet50(images[:,:,:,0:3]) # select 3 channels in a way that works only for consecutive channels
        images = tf.convert_to_tensor(images.numpy()[:,:,:,[0,2,4]]) # select 3 non-consecutive channels
        images = preprocess_input_resnet50(images)
        images = tf.dtypes.cast(images, np.float32)
        meta_data = tf.dtypes.cast(meta_data, np.float32)

        # trying with minimalistic meta where only use daytime_flag and clearsky
        meta_data = meta_data[:, -2:]

    elif 'double_pretrained_resnet' == args.model_name or 'resnet' == args.model_name:

        images = tf.dtypes.cast(images, np.float32)
        meta_data = tf.dtypes.cast(meta_data, np.float32)

        # meta_data is of shape (nb sample, 9) where the 9 features are:
        # [sin_month,cos_month,sin_minute,cos_minute, lat, lont, alt, daytime_flag, clearsky]
        # trying with minimalistic meta where only use daytime_flag and clearsky
        meta_data = meta_data[:, -2:]

    else:
        images = tf.keras.utils.normalize(images, axis=-1)

    return images, meta_data

def train_step(model, optimizer, meta_data, images, labels):
    images, meta_data = preprocess(images, meta_data)
    # Record the operations used to compute the loss, so that the gradient
    # of the loss with respect to the variables can be computed.
    with tf.GradientTape() as tape:
        y_pred = model(meta_data, images, training=True)
        loss = compute_loss(labels, y_pred)   
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if tf.equal(optimizer.iterations % 50, 0):
        # print('***pred', np.mean(y_pred, axis=0), 'label:', np.mean(labels, axis=0), 'nb prediction 0:', np.sum(y_pred <= 1))
        print("***predictions: ", y_pred[0].numpy(), "labels", labels[0].numpy())

    return loss


def train(model, optimizer, dataset, log_freq=1000):
    """
    Trains model on `dataset` using `optimizer`.
    """
    # Metrics are stateful. They accumulate values and return a cumulative
    # result when you call .result(). Clear accumulated values with .reset_states()
    avg_loss = metrics.Mean('loss', dtype=tf.float64)
    log_transform = False
    # Datasets can be iterated over like any other Python iterable.
    for (meta_data, images, labels) in dataset:
        if log_transform:
            images = tf.clip_by_value(
                images/255.0, clip_value_min=0, clip_value_max=5)
            labels = tf.clip_by_value(
                tf.math.log(labels+1), clip_value_min = 0, clip_value_max = 20)

        loss = train_step(model, optimizer, meta_data, images, labels)
        train_avg_loss(loss)

        if tf.equal(optimizer.iterations % log_freq, 0):
            print('step:', int(optimizer.iterations),
                  'loss:', train_avg_loss.result().numpy(),
                  'RMSE:', np.sqrt(train_avg_loss.result().numpy()))

            with train_summary_writer.as_default():
                tf.summary.scalar('RMSE', np.sqrt(avg_loss.result().numpy()), step=optimizer.iterations)
            avg_loss.reset_states()


def test(model, dataset):
    """
    Perform an evaluation of `model` on the examples from `dataset`.
    """
    valid_avg_loss = metrics.Mean('loss', dtype=tf.float64)

    for (meta_data, images, labels) in dataset:
        images, meta_data = preprocess(images, meta_data)

        y_pred = model(meta_data, images, training=False)
        valid_avg_loss(compute_loss(labels, y_pred))

    print('valid loss: {:0.4f} RMSE: {:0.2f}'.format(
        valid_avg_loss.result(), np.sqrt(valid_avg_loss.result().numpy())))
    return np.sqrt(valid_avg_loss.result().numpy())


def load_json(json_path):
    assert os.path.isfile(
        json_path), f"Invalid training configuration file: {json_path}"
    with open(json_path, "r") as tc:
        json_file = json.load(tc)
        return json_file

if __name__ == "__main__":
    print("Entering training python script.")

    # Arguments passed to training script
    parser = argparse.ArgumentParser()
    parser.add_argument("train_config", type=str,
                        help="Path of the training config file. This file contains ")
    # parser.add_argument("valid_config", type=str,
    #                     help="Path of the training config file. This file contains ")
    parser.add_argument("-s", "--seed", type=int, default=1234,
                        help="random seed for training")
    parser.add_argument("-n", "--num_epochs", type=int, default=100,
                        help="Number of epochs we want the model to train")
    parser.add_argument("-m", "--model_name", type=str, default="DummyModel",
                        help="To train a specific model, you can specify the name of your model. The ModelFactory will return the right model to train.\
                            The model name should be the modle class name. Example: 'DummyModel'. ")
    parser.add_argument("-u", "--user_config", type=str, default="",
                        help="Path to the JSON config file used to store user model/dataloader parameters")
    parser.add_argument("--scratch_dir", type=str, default=None,
                        help="Important for performance on the cluster!! If you want the files to be read fast, please set this variable.")
    parser.add_argument("--model_dir", type=str, default="./models",
                        help="Directory to save the checkpoints")
    parser.add_argument("--predict", action='store_true', default=False,
                        help="Predict with trained model or not")
    parser.add_argument("--use_cache", action='store_true', default=False,
                        help="Enable dataloader cache or not, default is False")
    parser.add_argument("--delete_checkpoints", action='store_true', default=False,
                        help="Delete previous checkpoints or not, by default is False")
    parser.add_argument("--load_checkpoints", action='store_true', default=False,
                        help="load previous checkpoints or not, by default is False")
    parser.add_argument("--save_best", action='store_true', default=False,
                        help="save best checkpoints or not, by default is False")
    parser.add_argument("--run_setting", type=str, default=None,
                        help="Summary of the run for tensorboard")
    args = parser.parse_args()

    if args.predict:
        print("Start predicting with existing model")
        if args.load_checkpoints is False:
            print("Please add flag --load_checkpoints!")


    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join('logs',args.model_name,current_time,'train') # 'logs/'+args.model_name+'/'+current_time+'/train'
    valid_log_dir = os.path.join('logs',args.model_name,current_time,'valid') #'logs/'+args.model_name+'/'+current_time+'/valid'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

    if args.run_setting:
        with train_summary_writer.as_default():
            tf.summary.text(name="Run_Settings_"+args.model_name, data=args.run_setting, step=0)
        with valid_summary_writer.as_default():
            tf.summary.text(name="Run_Settings_"+args.model_name, data=args.run_setting, step=0)


    print("Starting Training!")

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # Load configs
    train_json = load_json(args.train_config)
    # valid_json = load_json(args.valid_config)  

    user_config_json = None
    if args.user_config:
        with open(args.user_config, "r") as uc:
            user_config_json = json.load(uc)

    cache_dir = args.scratch_dir or os.getcwd()
    batch_size = train_json.get("batch_size") or 16
    buffer_size = train_json.get("buffer_size") or 500
    data_frame_path = extract_data_frame_path(train_json)
    stations, target_time_offsets = extract_station_offsets(train_json)

    # Init models, dataset and other vars for the training loop
    print("*******Create Model********")
    model_factory = ModelFactory(
        stations, target_time_offsets, args.user_config)
    model = model_factory.build(args.model_name)

    print("*******Create training dataset********")
    # if args.use_cache:
    #     train_ds = TrainingDataSet(data_frame_path, stations, train_json, user_config=user_config_json, scratch_dir=args.scratch_dir) \
    #         .prefetch(tf.data.experimental.AUTOTUNE) \
    #         .batch(batch_size) \
    #         .cache(cache_dir + "/tf_learn_cache") \
    #         .shuffle(buffer_size)
    #     valid_ds = TrainingDataSet(data_frame_path, stations, valid_json, user_config=user_config_json, scratch_dir=args.scratch_dir) \
    #         .prefetch(tf.data.experimental.AUTOTUNE) \
    #         .batch(batch_size) \
    #         .cache(cache_dir + "/tf_valid_cache") \
    #         .shuffle(buffer_size)
    # else:
    #     train_ds = TrainingDataSet(data_frame_path, stations, train_json, user_config=user_config_json, scratch_dir=args.scratch_dir) \
    #         .prefetch(tf.data.experimental.AUTOTUNE) \
    #         .batch(batch_size) \
    #         .shuffle(buffer_size)
    #     valid_ds = TrainingDataSet(data_frame_path, stations, valid_json, user_config=user_config_json, scratch_dir=args.scratch_dir) \
    #         .prefetch(tf.data.experimental.AUTOTUNE) \
    #         .batch(batch_size) \
    #         .shuffle(buffer_size)

    train_ds, valid_ds = solar_datasets()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005) #0.00003
    compute_loss = tf.keras.losses.MSE

    # Where to save checkpoints, tensorboard summaries, etc.
    checkpoint_dir = os.path.join(args.model_dir, args.model_name,'checkpoints')
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    best_checkpoint_prefix = os.path.join(checkpoint_dir, 'best-ckpt')

    # clear previous checkpoints for debug purpose
    if args.delete_checkpoints:
        apply_clean(checkpoint_dir)

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

    # Restore variables on creation if a checkpoint exists.
    if args.load_checkpoints:
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    lowest_valid_rmse = np.inf

    valid_avg_loss = metrics.Mean('loss', dtype=tf.float32)

    for i in range(args.num_epochs):       
        start = time.time()
        train(model, optimizer, train_ds, log_freq=100)
        end = time.time()
        print('Epoch #{} ({} total steps): {}sec RMSE: {}'.format(
            i + 1, int(optimizer.iterations), end - start, rmse))

        checkpoint.save(checkpoint_prefix)
        print('saved checkpoint.')

        valid_rmse = test(model, valid_ds)

        if args.save_best and valid_rmse < lowest_valid_rmse:
            checkpoint.save(best_checkpoint_prefix)
            print('saved best checkpoint.')

        with valid_summary_writer.as_default():
            tf.summary.scalar('RMSE', valid_rmse , step=optimizer.iterations)


    # export_path = os.path.join(MODEL_DIR, 'export')
    # tf.saved_model.save(model, export_path)
    # print('saved SavedModel for exporting.')
