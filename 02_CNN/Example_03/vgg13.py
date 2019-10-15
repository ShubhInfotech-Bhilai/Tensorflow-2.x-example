import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(100)


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(tf.squeeze(y), dtype=tf.int64)
    y = tf.one_hot(y, depth=100)
    return x, y


def gen_batch(x, y, batch_size):
    db = tf.data.Dataset.from_tensor_slices((x, y))
    db = db.map(preprocess).shuffle(x.shape[0]).batch(batch_size)
    return db


class VGG13_Model(tf.keras.Model):
    def __init__(self):
        super(VGG13_Model, self).__init__()

        layers_conv = [Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu),  # [b,32,32,3]
                       Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu),
                       MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),  # shape: [b,16,16,64]

                       Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu),
                       Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu),
                       MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),  # shape: [b,8,8,128]

                       Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu),
                       Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu),
                       MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),  # shape: [b,4,4,256]

                       Conv2D(filters=512, kernel_size=3, padding='same', activation=tf.nn.relu),
                       Conv2D(filters=512, kernel_size=3, padding='same', activation=tf.nn.relu),
                       MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),  # shape: [b,2,2,512]

                       Conv2D(filters=512, kernel_size=3, padding='same', activation=tf.nn.relu),
                       Conv2D(filters=512, kernel_size=3, padding='same', activation=tf.nn.relu),
                       MaxPool2D(pool_size=[2, 2], strides=2, padding='same')]  # shape: [b,1,1,512]
        self.seq_conv = tf.keras.Sequential(layers=layers_conv)
        layers_fc = [Dense(units=256, activation=tf.nn.relu),  # shape: [b,256]
                     Dense(units=128, activation=tf.nn.relu),  # shape: [b,256]
                     Dense(units=100, activation=None)]  # shape: [b,100]
        self.seq_fc = tf.keras.Sequential(layers_fc)

    def call(self, inputs, training=None):
        """
        只要一个操作被定义为是标准的tensorflow Layer类（含义__init__,call,build等方法）
        都可以通过Sequential 类来完成前向传播
        例如，此处的reshape 操作，其实可以通过Flatten() 这个Layerl类来完成，也就是上面的conv 部分和fc部分，通过一个Sequential就能实现整个前向传播
        如上一个示例Example_02

        :param inputs:
        :param training:
        :return:
        """

        conv_out = self.seq_conv(inputs)
        out = tf.reshape(conv_out, [-1, 512])
        out = self.seq_fc(out)
        return out


class VGG13():
    def __init__(self,
                 learning_rate=1e-4,
                 batch_size=128,
                 epochs=40):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.build_model()

    def build_model(self):
        self.model = VGG13_Model()
        self.model.compile(convtimizer=tf.keras.optimizers.Adam(lr=self.learning_rate),
                           loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def train(self, x_train, y_train, x_test, y_test):
        db_train = gen_batch(x_train, y_train, self.batch_size)
        db_test = gen_batch(x_test, y_test, self.batch_size)
        dir = 'ckpt/weights.ckpt'
        if os.path.exists(dir[:4]):
            self.model.load_weights(dir)
            print("Load existed weights successfully.")
            self.model.evaluate(db_test)
        try:
            self.model.fit(db_train, epochs=self.epochs, validation_data=db_test, validation_freq=2)
            self.model.summary()
            self.model.evaluate(db_test)
            self.model.save_weights(dir)
            print("Save weights successfully.")
        except KeyboardInterrupt:
            self.model.save_weights(dir)
            print("Save weights successfully.")


def main():
    (x, y), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    print("dataset info: x:{}, y:{}, x_test: {}, y_test: {}, min: {}, max:{}".
          format(x.shape, y.shape, x_test.shape, y_test.shape, x.min(), y.min()))

    # db_train = gen_batch(x, y, 64)
    # sample = next(iter(db_train))
    # print(sample[1])
    # print(sample[1].shape)

    model = VGG13()
    model.train(x, y, x_test, y_test)


#

if __name__ == '__main__':
    main()
