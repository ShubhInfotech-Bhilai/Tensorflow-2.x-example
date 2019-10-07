import tensorflow as tf
from tensorflow import keras
from train import preprocess, gen_batches
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
if __name__ == '__main__':
    _, (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    model = tf.keras.models.load_model('model.h5', compile=True)
    # model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
    #               loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    #               metrics=['accuracy'])
    db_test = gen_batches(x_test, y_test, 64)
    model.evaluate(db_test)

