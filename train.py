from __future__ import absolute_import, division, print_function, unicode_literals

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

from models.model_factory import ModelFactory
from dataloader.dataset import TrainingDataSet

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

def plot_loss(loss_values):
    plt.figure()

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
    parser.add_argument("--training", type=bool, default=True,
                        help="Enable training or not")
    parser.add_argument("--dont_use_cache", dest='dont_use_cache', action='store_true',
                        help="Add this flag if you don't want to use dataset cache")

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
    if not args.dont_use_cache:
        dataset = TrainingDataSet(data_frame_path, stations, train_json, user_config=user_config_json, scratch_dir=args.scratch_dir) \
            .prefetch(tf.data.experimental.AUTOTUNE) \
            .batch(batch_size) \
            .cache(cache_dir + "/tf_learn_cache") \
            .shuffle(buffer_size)
    else:
        dataset = TrainingDataSet(data_frame_path, stations, train_json, user_config=user_config_json, scratch_dir=args.scratch_dir) \
            .prefetch(tf.data.experimental.AUTOTUNE) \
            .batch(batch_size)

    train_loss_results = []
    train_accuracy_results = []
    is_training = args.training
    loss_fct = tf.keras.losses.MSE

    print("Model and dataset loaded, starting main training loop...!!")

    logging.basicConfig(filename='/project/cq-training-1/project1/teams/team07/result.log',level=logging.DEBUG)

    # main loop
    for epoch in range(args.num_epochs):
        datafetch_time = time.perf_counter()
        epoch_loss_avg = tf.keras.metrics.Mean()
        start_time = time.perf_counter()
        count = 0

        print("*******EPOCH %d start********" % (epoch+1))

        for iter_idx, (metas, images, targets) in enumerate(dataset):
            if count == 10:
                # print(f"Data Fetch time: {time.perf_counter() - datafetch_time}, for batch size: {metas.shape[0]}")
                count = 0
                
            
            with tf.GradientTape() as tape:
                y_ = model(metas, images)
                loss_value = loss_fct(y_true=targets, y_pred=y_)
                if iter_idx % 9 == 0:
                    print("********predicted value")
                    print(y_)
                    print(f"Batch loss: {loss_value}")
                
            # Track progress
            epoch_loss_avg(loss_value)  # Add current batch loss
            count += 1
            if count == 10:
                datafetch_time = time.perf_counter()

            if iter_idx % 9 == 0:
                print("epoch : %d , iter: %d,  epoch loss: %s" %
                      (epoch + 1, iter_idx + 1, loss_value))

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        logging.debug(train_loss_results)
        print(f"Epoch result: {epoch_loss_avg.result()}")
        print(f"Elapsed time for epoch: {time.perf_counter() - start_time}")
