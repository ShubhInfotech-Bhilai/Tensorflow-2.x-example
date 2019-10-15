import tensorflow as tf


class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()

        layers = [tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(3, 3), strides=stride, padding='same'),
                  tf.keras.layers.BatchNormalization(),
                  tf.keras.layers.Activation(activation='relu'),
                  tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(3, 3), strides=1, padding='same'),
                  tf.keras.layers.BatchNormalization()]
        self.sequential = tf.keras.Sequential(layers)  #

        if stride != 1:
            self.identity = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(1, 1), strides=stride)
        else:
            self.identity = lambda x: x

    def call(self, inputs, training=None):
        out = self.sequential(inputs)
        identity = self.identity(inputs)
        output = tf.keras.layers.add([out, identity])
        output = tf.nn.relu(output)
        return output


class ResNet(tf.keras.Model):
    def __init__(self, layer_dims, num_classes=100):
        super(ResNet, self).__init__()
        self.stem = tf.keras.Sequential(
            [tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1)),
             tf.keras.layers.BatchNormalization(),
             tf.keras.layers.Activation('relu'),
             tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same'),  # [b,32,32,64]

             self.build_res_block(64, layer_dims[0]),  # [b,32,32,64]
             self.build_res_block(128, layer_dims[1], stride=2),  # [b,16,16,128]
             self.build_res_block(256, layer_dims[2], stride=2),  # [b,8,8,256]
             self.build_res_block(512, layer_dims[3], stride=2),  # [b,4,4,512]
             tf.keras.layers.GlobalAveragePooling2D(),
             tf.keras.layers.Dense(units=num_classes)])

    def call(self, inputs, training=None):
        # print('training==================>', training)
        x = self.stem(inputs)
        return x

    def build_res_block(self, filter_num, num_res_blocks, stride=1):
        res_blocks = tf.keras.Sequential()
        res_blocks.add(BasicBlock(filter_num, stride))
        for _ in range(1, num_res_blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))
        return res_blocks


def resnet18():
    return ResNet([3, 4, 6, 3])
