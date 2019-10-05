import tensorflow as tf
from tensorflow import keras


def preprocess(x, y):  # 预处理
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int64)
    return x, y


(x, y), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
batch_size = 64
db_train = tf.data.Dataset.from_tensor_slices((x, y))
db_train = db_train.map(preprocess).shuffle(60000).batch(batch_size)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).shuffle(10000).batch(batch_size)

model = keras.Sequential([
    keras.layers.Dense(256, activation=tf.nn.relu),  # [b,784] => [b,256]
    keras.layers.Dense(128, activation=tf.nn.relu),  # [b,256] => [b,128]
    keras.layers.Dense(64, activation=tf.nn.relu),  # [b,128] => [b,64]
    keras.layers.Dense(32, activation=tf.nn.relu),  # [b,64] => [b,32]
    keras.layers.Dense(10),  # [b,32] => [b,10]
])
model.build(input_shape=[None, 28 * 28])
model.summary()
optimizer = tf.keras.optimizers.Adam(lr=1e-3)


def main():
    # train
    for epoch in range(5):
        for step, (x_batch, y_batch) in enumerate(db_train):
            x_batch = tf.reshape(x_batch, shape=[-1, 28 * 28])  # x_batch: [b,28,28] = > [b,784]
            y_batch = tf.one_hot(y_batch, depth=10)  # y_batch: [b] => [b,10]
            with tf.GradientTape() as tape:
                # [b,784] => [b,10]
                logits = model(x_batch)
                loss_mse = tf.reduce_mean(tf.losses.MSE(y_batch, logits))
                loss_cross_entropy = tf.reduce_mean(
                    tf.losses.categorical_crossentropy(y_batch, logits, from_logits=True))
            grads = tape.gradient(loss_cross_entropy, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print(epoch, step, 'loss: ', float(loss_cross_entropy), float(loss_mse))
        # test
        total_correct = 0
        for x, y in db_test:
            x = tf.reshape(x, [-1, 28 * 28])
            logits = model(x)
            prob = tf.nn.softmax(logits, axis=1)  # logits => probility, [b,10]
            pred = tf.argmax(prob, axis=1)  # [b,10] => [b]
            correct = tf.reduce_sum(tf.cast(tf.equal(pred, y), dtype=tf.int32))
            total_correct += int(correct)
        acc = total_correct / x_test.shape[0]
        print("Accuracy: {}".format(acc))


if __name__ == '__main__':
    main()
