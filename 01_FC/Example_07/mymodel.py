import tensorflow as tf


class MyDense(tf.keras.layers.Layer):
    def __init__(self, inp_dim, outp_dim, activation=None):
        super(MyDense, self).__init__()
        self.kernel = self.add_weight('w', [inp_dim, outp_dim])
        # self.bias = self.add_weight('b', [outp_dim])
        self.activation = activation

    def call(self, inputs, training=None):
        # out = tf.matmul(inputs, self.kernel) + self.bias
        out = tf.matmul(inputs, self.kernel)
        if self.activation is not None:
            return self.activation(out)
        return out


class MyNetwork(tf.keras.Model):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.fc1 = MyDense(32 * 32 * 3, 256, activation=tf.nn.relu)  # [b,784] => [b,256]
        self.fc2 = MyDense(256, 128, activation=tf.nn.relu)  # [b,256] => [b,128]
        self.fc3 = MyDense(128, 64, activation=tf.nn.relu)  # [b,128] => [b,64]
        self.fc4 = MyDense(64, 32, activation=tf.nn.relu)  # [b,64] => [b,32]
        self.fc5 = MyDense(32, 10)  # [b,32] => [b,10]

    def call(self, inputs, training=None):
        """

        :param inputs: [b, 32, 32, 3]
        :param training:
        :return:
        """
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x


