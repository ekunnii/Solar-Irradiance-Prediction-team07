import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, LSTM, GRU, Dropout
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet50 import preprocess_input
from utils.wrapper import TimeDistributed

class cnn_seq2seq(Model):
    def __init__(self, target_time_offsets):
        super(cnn_seq2seq, self).__init__()
        self.resnet50 = TimeDistributed(ResNet50(
            include_top=False, weights='imagenet', input_shape=(64, 64, 3)))
        self.avg_pool = TimeDistributed(GlobalAveragePooling2D())
        # nb of channels at the end of resnet + len(metas)
        self.d1 = Dense(16, activation="sigmoid")
        self.d2 = Dense(len(target_time_offsets), activation="linear")
        self.gru_encoder = GRU(units=256, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.attention = BahdanauAttention(10)        
        self.gru_decoder = GRU(units=256, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.dropout = Dropout(0.5)

    def input_transform(self, images):
        # if images.shape[1] != 6:
        #     return None
        #if pretrained, must use the same preprocess as when the model was trained, here preprocess of resnet
        # Images are 5D tensor [batch_size, past_images, image_size, image_size, channnel]
        batch_size = images.shape[0]
        image_size = images.shape[2]  # assume square images

        images = tf.reshape(images, [-1, image_size, image_size, 5])        
        images = tf.convert_to_tensor(images.numpy()[:,:,:,[0,2,4]])
        images = preprocess_input(images)
        images = tf.reshape(
            images, [batch_size, -1, image_size, image_size, 3])

        return images
    # def metas_embedding(self, metas):
    #     return metas = 

    def call(self, metas, images):
        assert not np.any(np.isnan(images))
        images = tf.dtypes.cast(images, np.float32)
        metas = tf.dtypes.cast(metas, np.float32)
        images = self.input_transform(images)
        # [sin_month,cos_month,sin_minute,cos_minute, lat, lont, alt, daytime_flag, clearsky]
        # use alt as station_idx
        metas = metas.numpy()
        metas[:, 6] /=1689
        metas = tf.convert_to_tensor(metas[:, [0,1,2,3,6,7]])
        
        x = self.resnet50(images)
        x = self.avg_pool(x)  # transform to (nb of sample, nb of channel)
        seq_outputs, last_hidden = self.gru_encoder(x)

        context_vector, attention_weights = self.attention(last_hidden, seq_outputs)

        # input vector to decoder has to be [batch, seq, vector]
        metas = tf.expand_dims(tf.concat([context_vector, metas], axis=-1), 1)

        # passing the concatenated vector to the GRU
        output, state = self.gru_decoder(metas)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        # output = self.dropout(output)
        output = self.d2(output)
        output = tf.clip_by_value(tf.math.exp(output), clip_value_min = 0, clip_value_max = 1500)

        return output

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights