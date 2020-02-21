import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, LSTM, TimeDistributed, ConvLSTM2D, BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet50 import preprocess_input


class double_cnn_lstm(Model):
    def __init__(self, target_time_offsets):
        super(double_cnn_lstm, self).__init__()
        # self.preprocess_input = TimeDistributed(preprocess_input())

        self.resnet50_1 = TimeDistributed(ResNet50(include_top=False, weights='imagenet', input_shape=(64, 64, 3)))
        self.resnet50_2 = TimeDistributed(ResNet50(include_top=False, weights='imagenet', input_shape=(64, 64, 3)))


        self.avg_pool = TimeDistributed(GlobalAveragePooling2D())

        self.d1 = TimeDistributed(Dense(500, activation='relu'))  # nb of channels at the end of a resnet  * 2 + len(metas)
        # self.d2 = Dense(2048 + 2, activation='relu')
        self.d2 = Dense(len(target_time_offsets), activation="relu")
        self.lstm1 = LSTM(units=128)

    def input_transform(images):
        if images.shape[1] != 6:
            return None

        # when pretrained, must use the same preprocess as when the model was trained, here preprocess of resnet
        # [batch, past_image, image_size, image_size, channel]
        batch_size = images.shape[0]
        image_size =  images.shape[2] # assume square images

        images = tf.reshape(images, [-1, image_size, image_size, 5])

        images_1 = tf.convert_to_tensor(images.numpy()[:,:,:,[0,2,4]])
        images_2 = tf.convert_to_tensor(images.numpy()[:,:,:,[1,2,3]])

        images_1 = preprocess_input(images_1)
        images_2 = preprocess_input(images_2)

        images_1 = tf.reshape(images_1, [batch_size, -1, image_size, image_size, 3])
        images_2 = tf.reshape(images_2, [batch_size, -1, image_size, image_size, 3])

        return images_1, images_2


    def call(self, metas, images):
        assert not np.any(np.isnan(images))

        images = tf.dtypes.cast(images, np.float32)
        images_1, images_2 = self.input_transform(images) # (Batch size, past images (6), weight, height, nb_channel)

        metas = tf.dtypes.cast(metas, np.float32)

        x_1 = self.resnet50_1(images_1)
        x_1 = self.avg_pool(x_1) #(batch size, past images, nb channels)

        x_2 = self.resnet50_2(images_2)
        x_2 = self.avg_pool(x_2)

        x = tf.concat([x_1,x_2], -1 ) #(batch size, past images, 2048*2)

        x = self.d1(x) #(batch_size, past images, 500)
        x = self.lstm1(x)
        x = tf.concat([x, metas], 1)
        x = self.d2(x)
        return x