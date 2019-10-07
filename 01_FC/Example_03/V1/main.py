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
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
              loss=tf.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


def main():
    # train

    model.fit(db_train, epochs=10, validation_data=db_test, validation_freq=1)
    model.evaluate(db_test)
    sample = next(iter(db_test))
    x, y = sample[0], sample[1]
    pred = model.predict(x)
    y = tf.argmax(y, axis=1)
    pred = tf.argmax(pred, axis=1)
    print("Labels:", y)
    print("Pred:", pred)



if __name__ == '__main__':
    main()
