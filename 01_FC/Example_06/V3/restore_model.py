import tensorflow as tf
from tensorflow import keras
from train import preprocess, gen_batches
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
if __name__ == '__main__':
    _, (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    imported_model = tf.saved_model.load('./Model/')
    model = imported_model.signatures['serving_default']

    image = tf.convert_to_tensor(x_test[:10].reshape(-1, 28 * 28)/255.,dtype=tf.float32)

    res = model(input_1=image)['output_1'] # 参数input_1可以通过报错来找到; model()返回的是一个字典

    # res = model(input=image)['output_1'] # 参数input_1可以通过报错来找到
    # TypeError: Expected argument names ['input_1'] but got values for ['input']. Missing: ['input_1'].
    # 需要的是input_1, 而我输入的却是input

    res = tf.nn.softmax(res)
    res = tf.argmax(res,axis = 1)
    print("pred:",res.numpy())
    print("label:",y_test[:10])


