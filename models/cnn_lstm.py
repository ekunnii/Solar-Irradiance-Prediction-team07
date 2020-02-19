import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, LSTM, TimeDistributed
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet50 import preprocess_input


# class cnn_lstm(Model):
#     def __init__(self, target_time_offsets):
#         super(cnn_lstm, self).__init__()
#         self.resnet50 = ResNet50(
#             include_top=False, weights='imagenet', input_shape=(64, 64, 3))
#         # self.flatten = Flatten()
#         self.avg_pool = GlobalAveragePooling2D()
#         # nb of channels at the end of resnet + len(metas)
#         self.d1 = Dense(2048+2, activation='relu')
#         self.d2 = Dense(len(target_time_offsets), activation="relu")
#         self.lstm1 = LSTM(units=128)

#     def call(self, metas, images):
#         assert not np.any(np.isnan(images))
#         images = tf.dtypes.cast(images, np.float32)
#         metas = tf.dtypes.cast(metas, np.float32)
#         # print(metas.shape)
#         # select only 3 channels because pre-trained on 3
#         image_embedding_seq = []
#         # TODO: image dimension [batch, past_image, image_size, image_size, channel]
#         for i in range(images.shape[1]):
#             x = tf.squeeze(images[:, i, ...])
#             x = preprocess_input(x[:, :, :, 0:3])
#             x = self.resnet50(x)
#             x = self.avg_pool(x)  # transform to (nb of sample, nb of channel)
#             # x = tf.concat([x, metas], 1)
#             # x = self.d1(x)
#             image_embedding_seq.append(x)
#         image_embedding_seq = tf.stack(image_embedding_seq)
#         image_embedding_seq = tf.transpose(image_embedding_seq, perm=[1, 0, 2])
#         x = self.lstm1(image_embedding_seq)
#         x = tf.concat([x, metas], 1)
#         x = self.d2(x)
#         return x


class cnn_lstm(Model):
    def __init__(self, target_time_offsets):
        super(cnn_lstm, self).__init__()
        # self.preprocess_input = TimeDistributed(preprocess_input())
        self.resnet50 = TimeDistributed(ResNet50(
            include_top=False, weights='imagenet', input_shape=(64, 64, 3)))
        # self.flatten = Flatten()
        self.avg_pool = TimeDistributed(GlobalAveragePooling2D())
        # nb of channels at the end of resnet + len(metas)
        self.d1 = Dense(2048+2, activation='relu')
        self.d2 = Dense(len(target_time_offsets), activation="relu")
        self.lstm1 = LSTM(units=128)

    def call(self, metas, images):
        assert not np.any(np.isnan(images))
        images = tf.dtypes.cast(images, np.float32)
        metas = tf.dtypes.cast(metas, np.float32)
        # select only 3 channels because pre-trained on 3
        # TODO: image dimension [batch, past_image, image_size, image_size, channel]
        # x = preprocess_input(images[:, :, :, 0:3])
        x = self.resnet50(images)
        x = self.avg_pool(x)  # transform to (nb of sample, nb of channel)
        # image_embedding_seq = tf.transpose(image_embedding_seq, perm=[1, 0, 2])
        x = self.lstm1(x)
        x = tf.concat([x, metas], 1)
        x = self.d2(x)
        return x
