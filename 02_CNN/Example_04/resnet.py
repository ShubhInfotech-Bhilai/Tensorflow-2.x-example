import tensorflow as tf


class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(3, 3), strides=stride, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation(activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(3, 3), strides=1, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        # 只要一个操作被定义为是标准的tensorflow Layer类（含义__init__,call,build等方法）
        # 都可以通过Sequential 类来完成前向传播
        # 例如，此处的第7到11行其实可以通过一个Sequential就能实现整个前向传播，见例Example_05的改写
        #
        if stride != 1:
            self.identity = tf.keras.Sequential()  #
            self.identity.add(tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(1, 1), strides=stride))
        else:
            self.identity = lambda x: x


    def call(self, inputs, training=None):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        identity = self.identity(inputs)
        output = tf.keras.layers.add([out, identity])
        output = tf.nn.relu(output)
        return output


class ResNet(tf.keras.Model):
    def __init__(self, layer_dims, num_classes=100):
        super(ResNet, self).__init__()
        self.stem = tf.keras.Sequential([tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1)),
                                         tf.keras.layers.BatchNormalization(),
                                         tf.keras.layers.Activation('relu'),
                                         tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')])
        self.layer1 = self.build_reblock(64, layer_dims[0])
        self.layer2 = self.build_reblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_reblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_reblock(512, layer_dims[3], stride=2)
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=num_classes)

    def call(self, inputs, training=None):
        # print('training==================>', training)
        x = self.stem(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x

    def build_reblock(self, filter_num, num_res_blocks, stride=1):
        res_blocks = tf.keras.Sequential()
        res_blocks.add(BasicBlock(filter_num, stride))
        for _ in range(1, num_res_blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))
        return res_blocks


def resnet18():
    return ResNet([2, 2, 2, 2])
