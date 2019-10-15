import tensorflow as tf
from resnet import resnet18

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(100)


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(tf.squeeze(y), dtype=tf.int64)
    y = tf.one_hot(y, depth=100)
    return x, y


def gen_batchs(x, y, batch_size):
    db = tf.data.Dataset.from_tensor_slices((x, y))
    db = db.map(preprocess).shuffle(x.shape[0]).batch(batch_size)
    return db


class ResNet18():
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

        self.model = resnet18()
        self.model.build(input_shape=(None, 32, 32, 3))
        self.model.summary()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                           loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def train(self, x_train, y_train, x_test, y_test):
        db_train = gen_batchs(x_train, y_train, self.batch_size)
        db_test = gen_batchs(x_test, y_test, self.batch_size)
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
    net = ResNet18()
    net.train(x, y, x_test, y_test)


if __name__ == '__main__':
    main()
