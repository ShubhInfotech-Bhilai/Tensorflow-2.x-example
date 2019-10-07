# Tensorflow-2.x-example
- #### 模型示例

  - ##### [01-普通前馈神经网络](./01_FC/README.md)

- #### 用法示例索引

  - ##### [001-全连接层用法](./01_FC/README.md)
  
    - `tf.keras.layers.Dense(),tf.keras.Sequential()`
    - `tf.losses.categorical_crossentropy(from_logits=Trye)`<==>`tf.nn.softmax_cross_entropy_with_logits()`
  
  - ##### [002-Tensorboard可视化](./01_FC/README.md)
  
    - `tf.summary.create_file_writer()`
  
  - ##### [003-通过Keras高级API进行网络训练](./01_FC/README.md)
  
    - `model.build(),model.compile(),model.fit(),model.evaluate()`
    - `tf.keras.losses.CategoricalCrossentropy()`
  
  - ##### [004-自定义Layer层并通过`keras.Sequential`进行训练](./01_FC/README.md)
  
  - ##### [005-自定义Layer层并通过自定义Model搭建模型](./01_FC/README.md)
  
  - ###### [006-模型的加载与保存]()
  
    - `model.save_weights(),model.save(),tf.saved_model`





# 引用仓库

##### https://github.com/dragen1860/TensorFlow-2.x-Tutorials