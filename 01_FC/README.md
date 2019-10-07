# 普通前馈神经网络

- #### 示例一：fashion_mnist图片分类 [Version1](./Example_01/V1/main.py)，[Version2](./Example_01/V2/main.py)

  使用最原始的方式来搭建网络

- #### 示例二：fashion_mnist图片分类+Tesorboard可视化 [Version1](./Example_02/V1/main.py)，[Version2](./Example_02/V2/main.py)

  加入Tensorboard可视化  [远程连接tensorboard](https://blog.csdn.net/The_lastest/article/details/94041583)

- #### 示例三：fashion_mnist图片分类 [Version1](./Example_03/V1/main.py)，[Version2](./Example_03/V2/main.py)

  通过`tf.keras`高级API(`compile,fit,predict`)来完成训练，并输出历史Loss值

- #### 示例四：自定义Layer层并通过`keras.Sequential`进行训练 [Version1](./Example_04/main.py)

  自定义一个Layer，然后加入到`keras.Sequential`列表中进行训练

- #### 示例五：自定义Layer层并通过自定义Model搭建模型 [Version1](./Example_05/main.py)

  分别自定义一个Layer层和一个Model类来完成模型的搭建与训练，这个Model类继承自`tf.keras.Model`，因此Model类同样拥有`model.fit(),model.compile()`等方法。

#### [返回主页](../README.md)



