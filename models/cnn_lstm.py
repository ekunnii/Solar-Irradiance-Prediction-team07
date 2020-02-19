import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, LSTM, TimeDistributed, ConvLSTM2D, BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.applications.resnet50 import preprocess_input


# class cnn_lstm(Model):
#     def __init__(self, target_time_offsets):
#         super(cnn_lstm, self).__init__()
#         self.flatten = Flatten()
#         self.d1 = Dense(30, activation='relu')
#         self.d2 = Dense(len(target_time_offsets), activation="relu")

#         self.conv1 = ConvLSTM2D(filters=64, kernel_size=(3, 3),
#                                 input_shape=(6, 64, 64, 5),
#                                 padding='same', return_sequences=True,
#                                 activation='relu')
#         self.BN = BatchNormalization()
#         self.conv2 = ConvLSTM2D(filters=64, kernel_size=(3, 3),
#                                 padding='same', return_sequences=True,
#                                 activation='relu')


#     def call(self, metas, images):
#         assert not np.any(np.isnan(images))
#         images = tf.dtypes.cast(images, np.float32)
#         metas = tf.dtypes.cast(metas, np.float32)
#         # select only 3 channels because pre-trained on 3
#         # TODO: image dimension [batch, past_image, image_size, image_size, channel]
#         # x = preprocess_input(images[:, :, :, 0:3])
#         x = self.conv1(images)
#         x = self.BN(x)
#         x = self.conv2(x)
#         x = self.BN(x)
#         x = self.flatten(x)
#         x = self.d1(x)
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

    def input_transform(images):
        if images.shape[1] != 6:
            return None
            #if pretrained, must use the same preprocess as when the model was trained, here preprocess of resnet
        images = tf.reshape(images, [-1, 64, 64, 5])     
        images = preprocess_input(images[:,:,:,0:3])
        images = tf.reshape(images, [32, -1, 64, 64, 3])

        return images


    def call(self, metas, images):
        assert not np.any(np.isnan(images))
        images = tf.dtypes.cast(images, np.float32)
        metas = tf.dtypes.cast(metas, np.float32)
        # images = self.input_transform(images)
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
