import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras import Model

class resnet(Model):
    def __init__(self, target_time_offsets):
        super(resnet, self).__init__()
        self.resnet50 = ResNet50(include_top=False, weights='imagenet', input_shape=(64, 64, 3))
        # self.flatten = Flatten()
        self.avg_pool = GlobalAveragePooling2D()
        self.d1 = Dense(2048+2, activation='relu') #nb of channels at the end of resnet + len(metas)
        self.d2 = Dense(len(target_time_offsets), activation="relu")

    def call(self, metas, images):
        assert not np.any(np.isnan(images))
        images = tf.dtypes.cast(images, np.float32)
        metas = tf.dtypes.cast(metas, np.float32)
        # select only 3 channels because pre-trained on 3
        x = images
        x = self.resnet50(x)
        x = self.avg_pool(x) # transform to (nb of sample, nb of channel)
        # x = self.flatten(x)
        x = tf.concat([x, metas], 1)
        x = self.d1(x)
        return self.d2(x)



## Create image of the architecture of the model
# tf.keras.utils.plot_model(
#     model,
#     to_file='model.png',
#     show_shapes=True,
#     show_layer_names=True,
#     rankdir='TB',
#     expand_nested=False,
#     dpi=96
# )


## Could transform the model with keras?
# resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(100, 100, 3))
# resnet = ResNet50(include_top=False, input_shape=(100, 100, 3))
# model = tf.keras.models.Sequential()
# model.add(resnet)
# model.add(Flatten())


## Example of how to directly use the pre-trained
# from tensorflow.keras.applications.resnet50 import ResNet50
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
# model = ResNet50(weights='imagenet')
# img_path = '../../elephant.jpg'
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
# preds = model.predict(x)
# # decode the results into a list of tuples (class, description, probability)
# # (one such list for each sample in the batch)
# print('Predicted:', decode_predictions(preds, top=3)[0])