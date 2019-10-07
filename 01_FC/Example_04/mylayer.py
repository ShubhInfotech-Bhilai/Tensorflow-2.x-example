import tensorflow as tf


class MyDense(tf.keras.layers.Layer):
    def __init__(self, inp_dim, outp_dim, activation=None):
        super(MyDense, self).__init__()
        self.kernel = self.add_weight('w', [inp_dim, outp_dim])
        self.bias =self.add_weight('b', [outp_dim])
        self.activation = activation

    def call(self, inputs, training=None):
        out = tf.matmul(inputs, self.kernel) + self.bias
        if self.activation is not None:
            return self.activation(out)
        return out
