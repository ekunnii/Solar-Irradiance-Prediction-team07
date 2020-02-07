import tensorflow as tf

class DummyModel(tf.keras.Model):

    def __init__(self, target_time_offsets):
        super(DummyModel, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(
            len(target_time_offsets), activation=tf.nn.softmax)

    def call(self, metas, images):
        x = self.dense1(self.flatten(self.flatten(images)))
        return self.dense2(x)