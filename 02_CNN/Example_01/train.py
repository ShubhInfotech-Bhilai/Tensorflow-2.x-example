import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

conv_layers = [
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu),  # [b,32,32,3]
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu),  # [b,32,32,64]
    tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),  # [b,32,32,64]

    tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu),
    tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),  # [b,32,32,128]

    tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu),
    tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),  # [b,32,32,256]

    tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation=tf.nn.relu),
    tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation=tf.nn.relu),
    tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')  # [b,32,32,512]
]

fc_layers = [
    tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=100, activation=None)
]


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.squeeze(tf.cast(y, dtype=tf.int64))
    y = tf.one_hot(y, depth=100)
    return x, y


def gen_batch(x, y, batch_size):
    db = tf.data.Dataset.from_tensor_slices((x, y))
    db = db.map(preprocess).shuffle(x.shape[0]).batch(batch_size)
    return db


def main():
    (x, y), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    print("dataset info: x.shape:{},y.shape:{},x_test.shape{},y_test.shape{}"
          .format(x.shape, y.shape, x_test.shape, y_test.shape))
    train_db = gen_batch(x, y, 64)
    test_db = gen_batch(x_test, y_test, 64)

    # sample = next(iter(train_db)) # 查看一个batch样本的信息
    # print(sample[1].shape)

    conv_net = tf.keras.Sequential(conv_layers)
    conv_net.build(input_shape=[None, 32, 32, 3])
    # conv_net.summary()

    fc_net = tf.keras.Sequential(fc_layers)
    fc_net.build(input_shape=[None, 512])

    # out = conv_net(sample[0]) # 通过喂入一个bath 来查看 模型的输出信息，在写代码的时候可以这样做
    # print(out.shape)  # [64,1,1,512]

    trainable_variables = conv_net.trainable_variables + fc_net.trainable_variables
    optimizer = tf.keras.optimizers.Adam(lr=1e-4)
    for epoch in range(500):
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                conv_out = conv_net(x)  # [b,1,1,512]
                conv_out = tf.reshape(conv_out, [-1, 512])
                logits = fc_net(conv_out)
                loss = tf.losses.categorical_crossentropy(y, logits, from_logits=True)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, trainable_variables)
            optimizer.apply_gradients(zip(grads, trainable_variables))
            if step % 100 == 0:
                print("Epoch:{}---step:{}---loss:{}".format(epoch, step, float(loss)))

            # test
        if epoch % 2 == 0:
            total_num = 0
            total_correct = 0
            for x, y in test_db:
                conv_out = conv_net(x)  # [b,1,1,512]
                conv_out = tf.reshape(conv_out, [-1, 512])
                logits = fc_net(conv_out)
                prob = tf.nn.softmax(logits, axis=1)
                pred = tf.argmax(prob, axis=1)
                correct = tf.cast(tf.equal(pred, tf.argmax(y, axis=1)), dtype=tf.float32)
                correct = tf.reduce_sum(correct)
                total_correct += correct
                total_num += x.shape[0]
            acc = total_correct / total_num
            print("Epoch:{}---accuracy:{}".format(epoch, acc))


if __name__ == '__main__':
    main()
    # conv_net = tf.keras.Sequential(conv_layer)
