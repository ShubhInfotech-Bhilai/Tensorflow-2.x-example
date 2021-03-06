import tensorflow as tf
from tensorflow import keras
from mymodel import MyModel
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def preprocess(x, y):  # 预处理
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [28 * 28])
    y = tf.cast(y, dtype=tf.int64)
    y = tf.one_hot(y, depth=10)
    return x, y


def gen_batches(x, y, batch_size):
    db = tf.data.Dataset.from_tensor_slices((x, y))
    db = db.map(preprocess).shuffle(x.shape[0]).batch(batch_size)
    return db


class MyNet():
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
        self.model = MyModel()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                           loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def train(self, x_train, y_train, x_test, y_test):
        db_train = gen_batches(x_train, y_train, self.batch_size)
        db_test = gen_batches(x_test, y_test, self.batch_size)
        self.model.fit(db_train, epochs=self.epochs, validation_data=db_test, validation_freq=2)
        self.model.summary()
        self.model.evaluate(db_test)
        # sample = next(iter(db_test))
        # x, y = sample[0], sample[1]
        # pred = self.model.predict(x)
        # y = tf.argmax(y, axis=1)
        # pred = tf.argmax(pred, axis=1)
        # print("Labels:", y)
        # print("Pred:", pred)


def main():
    (x, y), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    net = MyNet()
    net.train(x, y, x_test, y_test)


if __name__ == '__main__':
    main()
