# MNIST

## tensorboard
`python3 -m tensorboard.main --logdir=./v1/log`

## dataset
dataset is downloaded from http://yann.lecun.com/exdb/mnist/

[数据集的格式](https://blog.csdn.net/wspba/article/details/54311566)  
数据集是write('rb') 的file，然后还压缩了一下下，用gzip.GzipFile 打开
文件头几位包含特殊用途  

    Training set image file:  
    byte_offset value  description  
    0000        2051   magic number  
    0004        60000  number of images  
    0008        28     number of rows  
    0012        28     number of columns  
    0016        ?      pixel  
    0017        ?      pixel  
    ...

    training set label file:  
    byte_offset value  description  
    0000        2049   magic number  
    0004        60000  number of items  
    0008        ?      label  
    0009        ?      label  
    ...

## note

首先这是个多分类问题，0-9种数字，所以one-hot + [softmax](http://ufldl.stanford.edu/wiki/index.php/Softmax%E5%9B%9E%E5%BD%92)（可以理解为一个可导的函数，先由h函数归一化，再由J函数放大；用于表示互斥的多分类问题）  
然后是网络结构：

### network v1: linear classifier (1-layer NN)

先用一层的网络，即 W * x，其中W是模型参数，x是输入图像向量。
> test set accuracy: 0.9213

### network v2: Multi-layer networks (2-layer NN)

#### 2-layer NN
1)用softmax, 128 HU(hidden unit), cross-entropy
`self.y = tf.nn.softmax(tf.matmul(self.h1, self.w2) + self.b2, name='output_layer')`
```
step 120000 loss: 2.382  
     -> Training set error_rate: 0.0108  
     -> Test set error_rate: 0.0369  
```
2)用softmax(sigmoid), 128 HU(hidden unit), cross-entropy
```
self.h2 = tf.nn.sigmoid(tf.matmul(self.h1, self.w2) + self.b2)
self.y = tf.nn.softmax(self.h2, name='output_layer')
```
```
step 120000 loss: 99.614  
     -> Training set error_rate: 0.0591  
     -> Test set error_rate: 0.0640  
```
3)用sigmoid, 128 HU(hidden unit), cross-entropy
`self.y = tf.nn.sigmoid(tf.matmul(self.h1, self.w2) + self.b2)`
效果极差
```
step 120000 loss: 0.002  
     -> Training set error_rate: 0.7979  
     -> Test set error_rate: 0.7989  
```
- 值得注意的是，网络里面的变量要用随机数初始化，不然无法收敛
`tf.Variable(tf.random_uniform([784, 128], -1, 1)`
- 另外发现用以下随机数初始化，收敛更快，而且准确率会变高
`tf.Variable(tf.truncated_normal(shape=[], 0, 0.1))`

4)用softmax, 300 HU(hidden unit), cross-entropy
```
step 300000 loss: 0.128  
     -> Training set error_rate: 0.0000  
     -> Test set error_rate: 0.0304  
     -> Test set error_rate: 0.0304  
```
5)用softmax, 1000 HU, cross-entropy
```
step 150000 loss: 0.180207  
     -> Training set error_rate: 0.000000  
     -> Test set error_rate: 0.037500  
```
6)用softmax, 300 HU, MSE (收敛很慢)
```
step 7000000 loss: 0.005344  
     -> Training set error_rate: 0.003855  
     -> Test set error_rate: 0.034500  
```
```
step 10020000 loss: 0.000043  
     -> Training set error_rate: 0.003836  
     -> Test set error_rate: 0.034400  
```
#### 3-layer NN

1)用softmax, 500 + 150 sigmoid HU, cross-entropy (best error_rate: 0.018500)
```
- step 500000 loss: 0.039671  
     -> Training set error_rate: 0.000000  
     -> Test set error_rate: 0.037700
```
with `tf.truncated_normal([], 0, 0.1)`
```
step 120000 loss: 0.633766  
     -> Training set error_rate: 0.001873  
     -> Test set error_rate: 0.020600  
```
```
step 280000 loss: 0.162379  
     -> Training set error_rate: 0.000000  
     -> Test set error_rate: 0.018500  
```
1)用softmax, 500 + 150 ReLU HU, cross-entropy (best error_rate: 0.019500)  
收敛更快
with `tf.truncated_normal([], 0, 0.1)`
```
step 100000 loss: 0.019779  
     -> Training set error_rate: 0.000000  
     -> Test set error_rate: 0.019800    
```
otherwise 几乎收敛极慢


**全连接网络的缺点**：  
1）对于要识别的手写体的大小、倾斜、图片中的位置等干扰不敏感  
2）局部特征(local features)不明显  

### network v3: convolutional neural networks





---

参考文章

[1.tensorflow入门-mnist手写数字识别](https://geektutu.com/post/tensorflow-mnist-simplest.html)  
[2.Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)  
[3.Multi-column Deep Neural Networks for Image Classification](https://arxiv.org/pdf/1202.2745.pdf)  

