import tensorflow as tf
from tensorflow import keras
import datetime
from matplotlib import pyplot as plt
import io


def preprocess(x, y):  # 预处理
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int64)
    return x, y


def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image


def image_grid(images):
    figure = plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1, title='name')
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
    return figure


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

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)

sample_img = next(iter(db_train))[0]
sample_img = sample_img[0]
sample_img = tf.reshape(sample_img,[1,28,28,1])
with summary_writer.as_default():
    tf.summary.image("Training sampel:",sample_img,step=0)

def main():
    # train
    global_step = 0
    for epoch in range(50):
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

            with summary_writer.as_default():
                tf.summary.scalar('train-loss',float(loss_cross_entropy),step =global_step )
                global_step += 1

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

        test_images = next(iter(db_train))[0][:25]
        test_images = tf.reshape(test_images, [-1, 28, 28, 1])
        with summary_writer.as_default():
            tf.summary.scalar('test-acc',float(acc),step=epoch)
            tf.summary.image('test-onebyone-images:',test_images,max_outputs=25,step=epoch)

            test_images = tf.reshape(test_images,[-1,28,28])
            figure = image_grid(test_images)
            tf.summary.image('test-imgaes:',plot_to_image(figure),step=epoch)



if __name__ == '__main__':
    main()
