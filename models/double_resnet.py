import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet50 import preprocess_input  as preprocess_input_resnet50

class double_resnet(Model):
    def __init__(self, target_time_offsets):
        super(double_resnet, self).__init__()
        self.resnet50_1 = ResNet50(include_top=False, weights='imagenet', input_shape=(64, 64, 3))
        self.resnet50_2 = ResNet50(include_top=False, weights='imagenet', input_shape=(64, 64, 3))

        self.avg_pool = GlobalAveragePooling2D()
        self.d1 = Dense(2048, activation='relu') #nb of channels at the end of a resnet  * 2 + len(metas)
        self.d2 = Dense(1000, activation='relu')
        self.d3 = Dense(len(target_time_offsets), activation="relu")

    def call(self, metas, images):

        images_1 = tf.convert_to_tensor(images.numpy()[:,:,:,[0,2,4]])
        images_2 = tf.convert_to_tensor(images.numpy()[:,:,:,[1,2,3]])
        images_1 = preprocess_input_resnet50(images_1)
        images_2 = preprocess_input_resnet50(images_2)

        x_1 = self.resnet50_1(images_1)
        x_1 = self.avg_pool(x_1)

        x_2 = self.resnet50_2(images_2)
        x_2 = self.avg_pool(x_2)

        x = tf.concat([x_1, x_2, metas], 1)
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)