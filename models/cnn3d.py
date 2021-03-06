import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv3D, MaxPool3D, ZeroPadding3D, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
import os

class cnn3d(tf.keras.Model):
    def __init__(self, target_time_offsets):

        # deltatime to try: -30 min, -1h, -1h30, -2h, -3h, -4h, -5h
        super(cnn3d, self).__init__()
        self.conv1 = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', activation='relu', input_shape=(7, 64, 64, 5))
        self.conv2 = Conv3D(filters=64, kernel_size=(4, 4, 4), padding='same', activation='relu')
        self.conv3 = Conv3D(filters=32, kernel_size=(2, 2, 2), padding='same', activation='relu')
        self.maxpool1 = MaxPool3D(pool_size=(2, 2, 2), padding='same')
        self.flatten = Flatten()
        self.d1 = Dense(264, activation='relu')
        self.d2 = Dense(len(target_time_offsets), activation="linear")

    def call(self, metas, images):
        assert not np.any(np.isnan(images))
        x = self.conv1(images)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = self.maxpool1(x)
        x = self.flatten(x)
        x = tf.concat(axis=1,values=[x, metas])
        x = self.d1(x)
        return self.d2(x)

    def load_config(self, model, user_config):
        cnn3d_config = user_config.get("cnn3d")
        model_path = cnn3d_config.get("model_path")
        assert os.path.exists(model_path), f"Can't find model path: {model_path}"

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        checkpoint.restore(tf.train.latest_checkpoint(model_path))
        return model

