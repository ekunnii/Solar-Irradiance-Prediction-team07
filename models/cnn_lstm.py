import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, LSTM, ConvLSTM2D, BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
from utils.wrapper import TimeDistributed

class cnn_lstm(Model):
    def __init__(self, target_time_offsets):
        super(cnn_lstm, self).__init__()
        self.resnet50 = TimeDistributed(ResNet50(include_top=False, weights='imagenet', input_shape=(64, 64, 3)))
        self.avg_pool = TimeDistributed(GlobalAveragePooling2D())
        self.d1 = TimeDistributed(Dense(1024, activation='relu'))
        self.d1_1 = TimeDistributed(Dense(512, activation='relu'))
        self.d2 = Dense(32, activation='relu')
        self.d3 = Dense(len(target_time_offsets), activation="relu")
        self.lstm1 = LSTM(units=128)

    def input_transform(self, images):

        #if pretrained, must use the same preprocess as when the model was trained, here preprocess of resnet
        # [batch, past_image, image_size, image_size, channel]
        batch_size = images.shape[0]
        image_size =  images.shape[2] #assume square images
        images = tf.reshape(images, [-1, image_size, image_size, 5])
        images = tf.convert_to_tensor(images.numpy()[:,:,:,[0,2,4]])
        images = preprocess_input_resnet50(images)
        images = tf.reshape(images, [batch_size, -1, image_size, image_size, 3])

        return images


    def call(self, metas, images):
        images_ = tf.dtypes.cast(images, np.float32)
        metas_ = tf.dtypes.cast(metas, np.float32)

        images_ = self.input_transform(images_)
        metas_ = tf.convert_to_tensor(metas_.numpy()[:, [0,1,2,3,4,5,6,8]])

        x = self.resnet50(images_)
        x = self.avg_pool(x)
        x = self.d1(x)
        x = self.d1_1(x)
        x = self.lstm1(x)
        x = tf.concat([x, metas_], 1)
        x = self.d2(x)
        return self.d3(x)
