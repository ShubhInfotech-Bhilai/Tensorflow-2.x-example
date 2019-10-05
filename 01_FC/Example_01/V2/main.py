import tensorflow as tf
from tensorflow import keras


def preprocess(x, y):  # 预处理
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int64)
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
                 epoches=10,
                 ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_class = num_class
        self.epoches = epoches
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
        self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)

    def train(self, x_train, y_train, x_test, y_test):
        db_train = gen_batches(x_train, y_train, self.batch_size)
        db_test = gen_batches(x_test, y_test, self.batch_size)
        last_epoch_loss = -1.0
        for epoch in range(self.epoches):
            total_loss = 0
            for step, (x_batch, y_batch) in enumerate(db_train):
                x_batch = tf.reshape(x_batch, shape=[-1, 28 * 28])  # x_batch: [b,28,28] = > [b,784]
                y_batch = tf.one_hot(y_batch, depth=10)  # y_batch: [b] => [b,10]
                with tf.GradientTape() as tape:
                    # [b,784] => [b,10]
                    logits = self.model(x_batch)
                    loss_cross_entropy = tf.reduce_mean(
                        tf.losses.categorical_crossentropy(y_batch, logits, from_logits=True))
                    total_loss += float(loss_cross_entropy)

                grads = tape.gradient(loss_cross_entropy, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                if step % 100 == 0:
                    print("Epoch:{}/{}===Last epoch total loss on train {:.4}".format(epoch, self.epoches,
                                                                                      last_epoch_loss))
            total_correct = 0
            last_epoch_loss = total_loss
            for x, y in db_test:
                x = tf.reshape(x, [-1, 28 * 28])
                logits = self.model(x)
                prob = tf.nn.softmax(logits, axis=1)  # logits => probility, [b,10]
                pred = tf.argmax(prob, axis=1)  # [b,10] => [b]
                correct = tf.reduce_sum(tf.cast(tf.equal(pred, y), dtype=tf.int32))
                total_correct += int(correct)
            acc = total_correct / x_test.shape[0]
            print("Epoch:{}/{}===Accuracy:{}===Current epoch total loss on train {:.4f}".
                  format(epoch, self.epoches, acc, total_loss))


def main():
    (x, y), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    net = MyNet()
    net.train(x, y, x_test, y_test)


if __name__ == '__main__':
    main()
