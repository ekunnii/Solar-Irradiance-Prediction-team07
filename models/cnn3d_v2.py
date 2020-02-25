import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv3D, MaxPool3D, ZeroPadding3D, Concatenate, LeakyReLU, BatchNormalization, ZeroPadding3D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
import os

# based off https://github.com/OValery16/Tutorial-about-3D-convolutional-network/blob/master/model.py
class cnn3d_v2(tf.keras.Model):
    def __init__(self, target_time_offsets):

        # deltatime to try: -30 min, -1h, -1h30, -2h, -3h, -4h, -5h
        super(cnn3d, self).__init__()
        self.conv_layer1 = self._make_conv_layer(32, input_shape=[6, 64, 64, 5])
        self.conv_layer2 = self._make_conv_layer(64)
        self.conv_layer3 = self._make_conv_layer(124)
        self.conv_layer4 = self._make_conv_layer(256)
        self.conv_layer5 = Conv3D(filters=256, kernel_size=(1, 3, 3), padding="valid")

        self.flatten = Flatten()
        self.fc5 = Dense(256)
        self.relu = LeakyReLU(alpha=0.01)
        self.batch0 = BatchNormalization()
        self.drop = Dropout(0.15)        
        self.fc6 = Dense(124)
        # relu here
        self.batch1 = BatchNormalization()
        # drop here
        self.fc7 = Dense(len(target_time_offsets), activation="relu")

    def _make_conv_layer(self, out_c, input_shape=None):
        conv_layer = tf.keras.Sequential()
        if input_shape:
            conv_layer.add(ZeroPadding3D(padding=(1, 1, 1))) # manual padding
            conv_layer.add(Conv3D(filters=out_c, kernel_size=(2, 3, 3), padding="valid", input_shape=input_shape))
            conv_layer.add(LeakyReLU(alpha=0.01))
            #conv_layer.add(ZeroPadding3D(padding=(1, 1, 1))) # manual padding
            conv_layer.add(Conv3D(filters=out_c, kernel_size=(2, 3, 3), padding="same"))
            conv_layer.add(LeakyReLU(alpha=0.01))
            conv_layer.add(MaxPool3D(pool_size=(2, 2, 2)))
        else:
            conv_layer.add(ZeroPadding3D(padding=(1, 1, 1))) # manual padding
            conv_layer.add(Conv3D(filters=out_c, kernel_size=(2, 3, 3), padding="valid"))
            conv_layer.add(LeakyReLU(alpha=0.01))
            #conv_layer.add(ZeroPadding3D(padding=(1, 1, 1))) # manual padding
            conv_layer.add(Conv3D(filters=out_c, kernel_size=(2, 3, 3), padding="same"))
            conv_layer.add(LeakyReLU(alpha=0.01))
            conv_layer.add(MaxPool3D((2, 2, 2)))
        return conv_layer

    def call(self, metas, images):
        assert not np.any(np.isnan(images))
        x = self.conv_layer1(images)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.conv_layer5(x)

        x = self.flatten(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.batch0(x)
        x = self.drop(x)
        x = self.fc6(x)
        x = self.relu(x)
        x = self.batch1(x)
        x = self.drop(x)
        x = self.fc7(x)

        return x

    def load_config(self, model, user_config):
        cnn3d_config = user_config.get("cnn3d")
        model_path = cnn3d_config.get("model_path")
        assert os.path.exists(model_path), f"Can't find model path: {model_path}"

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        checkpoint.restore(tf.train.latest_checkpoint(model_path))
        return model

