import tensorflow as tf
import os
from mymodel import MyNetwork


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def preprocess(x, y):
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1
    x = tf.reshape(x, [32 * 32 * 3])  # 这里没有batch的维度
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(tf.squeeze(y), depth=10)
    return x, y


def gen_batch(x, y, batch_size):
    db = tf.data.Dataset.from_tensor_slices((x, y))
    db = db.map(preprocess).shuffle(x.shape[0]).batch(batch_size)
    return db


class MyModel():
    def __init__(self,
                 learning_rate=1e-3,
                 batch_size=64,
                 num_class=10,
                 epochs=5
                 ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_class = num_class
        self.epochs = epochs
        self.build_model()

    def build_model(self):
        self.model = MyNetwork()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                           loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def train(self, x_train, y_train, x_test, y_test):
        db_train = gen_batch(x_train, y_train, self.batch_size)
        db_test = gen_batch(x_test, y_test, self.batch_size)
        dir = 'ckpt/weights.ckpt'
        if os.path.exists(dir[:4]):
            self.model.load_weights(dir)
            print("load weights successfully")
            self.model.evaluate(db_test)

        self.model.fit(db_train, epochs=self.epochs, validation_data=db_test, validation_freq=2)
        self.model.summary()
        self.model.evaluate(db_test)
        self.model.save_weights(dir)
        print("save weights successfully")


def main():
    (x, y), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    print("dataset info: x:{}, y:{}, x_test: {}, y_test: {}, min: {}, max:{}".
          format(x.shape, y.shape, x_test.shape, y_test.shape, x.min(), y.min()))
    my_model = MyModel()
    my_model.train(x, y, x_test, y_test)


if __name__ == '__main__':
    main()
