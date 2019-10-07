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

- #### 示例六：模型的保存与加载的三种方法 [Version1](./Example_06/V1)，[Version2](./Example_06/V2)，[Version3](./Example_06/V3)

  模型的保存于加载一共有三种模式：

  - `save/load weights`：最轻量级，只保存网络的参数，其它状态通过不管，适用于有源代码的情况
  - `save/load entire model`：最粗暴，保存所有参数级状态，可以完美进行恢复
  - `saved_model`：一种保存模型的通用格式，跟pytorch对应的ONNX一样，可直接将模型拿去部署而不需要源代码。例如：用C++来解析这一模型完成部署。

#### [返回主页](../README.md)



