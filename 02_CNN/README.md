# 卷积神经网络

- #### 示例一：[VGG13网络cifar100图片分类](./Example_01/train.py)

  通过两个`tf.keras.Sequential`来分别完成卷积部分和全连接部分，然后通过`tf.GradientTape`求导完成整个网络过程的训练。

- #### 示例二：[VGG13网络cifar100图片分类](Example_02/vgg13.py)

  通过已有的`Conv2D(),Flatten(),Dense()`构造一个`tf.keras.Sequential`来完成整个网络的前向传播过程，然后通过`model.fit`来完成整个网络的训练。**一般需要实现的网络所用到的网络层在tensorflow中已经被实现（或者自己按照tensorflow的接口标准实现），则可以通过这种方法来实现网络的构建**

- #### 示例三：[VGG13网络cifar100图片分类](Example_03/vgg13.py)

  通过自定义一个`tf.keras.Model`来实现整个网络的前向传播过程，然后通过`model.fit`来完成整个网络的训练。**在设计自己的网络时，通常采用这一方法。**

  

