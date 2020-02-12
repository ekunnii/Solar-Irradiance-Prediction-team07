import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import Model
import numpy as np


class cnn2d(Model):
    def __init__(self, target_time_offsets):
        super(cnn2d, self).__init__()
        self.conv1 = Conv2D(
            32, (3, 3), activation='relu', input_shape=(64, 64, 5))
        self.conv2 = Conv2D(64, (3, 3), activation='relu')
        self.conv3 = Conv2D(64, (3, 3), activation='relu')
        self.maxpool = MaxPooling2D((2, 2))
        self.flatten = Flatten()
        self.d1 = Dense(64, activation='relu')
        self.d2 = Dense(len(target_time_offsets), activation="linear")

    def call(self, metas, images):
        assert not np.any(np.isnan(images))
        x = self.conv1(images)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


if __name__ == "__main__":
    cnn2d_model = cnn2d(target_time_offsets=[1, 2, 3, 4])
