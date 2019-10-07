import tensorflow as tf
from tensorflow import keras
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
        self.model = None
        self.build_model()

    def build_model(self):
        self.model = keras.Sequential([
            keras.layers.Dense(256, activation=tf.nn.relu),  # [b,784] => [b,256]
            keras.layers.Dense(128, activation=tf.nn.relu),  # [b,256] => [b,128]
            keras.layers.Dense(64, activation=tf.nn.relu),  # [b,128] => [b,64]
            keras.layers.Dense(32, activation=tf.nn.relu),  # [b,64] => [b,32]
            keras.layers.Dense(10),  # [b,32] => [b,10]
        ])
        self.model.build(input_shape=[None, 28 * 28])
        self.model.summary()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                           loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def train(self, x_train, y_train, x_test, y_test):
        db_train = gen_batches(x_train, y_train, self.batch_size)
        db_test = gen_batches(x_test, y_test, self.batch_size)
        self.model.fit(db_train, epochs=self.epochs, validation_data=db_test, validation_freq=2)
        self.model.evaluate(db_test)
        self.model.save('model.h5')
        print("model saved successfully", end='\n\n')
        del self.model




def main():
    (x, y), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    net = MyNet()
    net.train(x, y, x_test, y_test)



if __name__ == '__main__':
    main()
