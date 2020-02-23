import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, LSTM
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet50 import preprocess_input
from utils.wrapper import TimeDistributed

class cnn_lstm(Model):
    def __init__(self, target_time_offsets):
        super(cnn_lstm, self).__init__()
        # self.preprocess_input = TimeDistributed(preprocess_input())
        self.resnet50 = TimeDistributed(ResNet50(
            include_top=False, weights='imagenet', input_shape=(64, 64, 3)))
        # self.flatten = Flatten()
        self.avg_pool = TimeDistributed(GlobalAveragePooling2D())
        # nb of channels at the end of resnet + len(metas)
        self.d1 = TimeDistributed(Dense(256, activation='relu'))
        self.d2 = Dense(len(target_time_offsets), activation="relu")
        self.lstm1 = LSTM(units=128)

    def input_transform(self, images):
        # if images.shape[1] != 6:
        #     return None
        #if pretrained, must use the same preprocess as when the model was trained, here preprocess of resnet
        # Images are 5D tensor [batch_size, past_images, image_size, image_size, channnel]
        batch_size = images.shape[0]
        image_size = images.shape[2]  # assume square images

        images = tf.reshape(images, [-1, image_size, image_size, 5])
        images = preprocess_input(images[:, :, :, 0:3])
        images = tf.reshape(
            images, [batch_size, -1, image_size, image_size, 3])

        return images

    def call(self, metas, images):
        assert not np.any(np.isnan(images))
        images = tf.dtypes.cast(images, np.float32)
        metas = tf.dtypes.cast(metas, np.float32)
        images = self.input_transform(images)

        x = self.resnet50(images)
        x = self.avg_pool(x)  # transform to (nb of sample, nb of channel)
        x = self.d1(x)
        x = self.lstm1(x)
        x = tf.concat([x, metas], 1)
        x = self.d2(x)
        return x
