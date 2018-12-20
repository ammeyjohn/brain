#-*- coding:utf-8 -*_


#加载数据


import tensorflow as tf
from dataset import load_data

# 输入数据
# from tensorflow.examples.tutorials.mnist import input_data
#TensorFlow 自带，用来下载并返回 mnist 数据。可以自己下载 mnist数据后，存放到指定目录，我这里是 /tmp/data 目录。
#其实如果没有下载数据，TensorFlow 也会帮你自动下载 mnist 数据存放到你指定的目录当中。
#mnist 数据下载地址：http://yann.lecun.com/exdb/mnist/
# mnist = input_data.read_data_sets("./datasets/mnist", one_hot=True)

input_shape = ( 227, 227 )

# 定义网络的超参数
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 5

# 定义网络的参数
# n_input = 784 # 输入的维度 (img shape: 28*28)
n_classes = 5 # 标记的维度 (0-9 digits)
dropout = 0.75 # Dropout的概率，输出的可能性

# 输入占位符
x = tf.placeholder(tf.float32, [None, input_shape[0], input_shape[0], 3])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


#构建网络模型


# 定义卷积操作
def conv2d(name,x, W, b, strides=1, padding='SAME'):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x,name=name)  # 使用relu激活函数

# 定义池化层操作
def maxpool2d(name,x, m=3, k=2, padding='VALID'):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, m, m, 1], strides=[1, k, k, 1],
                          padding=padding,name=name)     #最大值池化

# 规范化操作
def norm(name, l_input, lsize=2):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=2e-05,
                     beta=0.75, name=name)

# 定义所有的网络参数
weights = {
    'wc1': tf.Variable(tf.random_normal([11, 11, 3, 96])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384])),
    'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384])),
    'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256])),
    'wd1': tf.Variable(tf.random_normal([6*6*256, 4096])),
    'wd2': tf.Variable(tf.random_normal([4096, 4096])),
    'out': tf.Variable(tf.random_normal([4096, n_classes]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([96])),
    'bc2': tf.Variable(tf.random_normal([256])),
    'bc3': tf.Variable(tf.random_normal([384])),
    'bc4': tf.Variable(tf.random_normal([384])),
    'bc5': tf.Variable(tf.random_normal([256])),
    'bd1': tf.Variable(tf.random_normal([4096])),
    'bd2': tf.Variable(tf.random_normal([4096])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def _print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

#定义 Alexnet 网络模型

# (self.feed('data')
#         .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
#         .lrn(2, 2e-05, 0.75, name='norm1')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
#         .conv(5, 5, 256, 1, 1, group=2, name='conv2')
#         .lrn(2, 2e-05, 0.75, name='norm2')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
#         .conv(3, 3, 384, 1, 1, name='conv3')
#         .conv(3, 3, 384, 1, 1, group=2, name='conv4')
#         .conv(3, 3, 256, 1, 1, group=2, name='conv5')
#         .fc(4096, name='fc6')
#         .fc(4096, name='fc7')
#         .fc(1000, relu=False, name='fc8')
#         .softmax(name='prob'))

# 定义整个网络
def alex_net(x, weights, biases, dropout):
    # 向量转为矩阵 Reshape input picture
    # x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # 第一层卷积
    conv1 = conv2d('conv1', x, weights['wc1'], biases['bc1'], strides=4, padding='VALID')
    norm1 = norm('norm1', conv1, lsize=2)
    pool1 = maxpool2d('pool1', norm1, m=3, k=2)
    _print_activations(norm1)

    # 第二层卷积
    conv2 = conv2d('conv2', pool1, weights['wc2'], biases['bc2'])
    norm2 = norm('norm2', conv2, lsize=2)
    pool2 = maxpool2d('pool2', norm2, m=3, k=2)
    _print_activations(norm2)

    # 第三层卷积
    conv3 = conv2d('conv3', pool2, weights['wc3'], biases['bc3'])
    _print_activations(conv3)

    # 第四层卷积
    conv4 = conv2d('conv4', conv3, weights['wc4'], biases['bc4'])
    _print_activations(conv4)

    # 第五层卷积
    conv5 = conv2d('conv5', conv4, weights['wc5'], biases['bc5'])
    pool5 = maxpool2d('pool5', conv5, m=3, k=2)
    _print_activations(pool5)

    # 全连接层1
    fc1 = tf.reshape(pool5, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 =tf.add(tf.matmul(fc1, weights['wd1']),biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # dropout
    fc1=tf.nn.dropout(fc1,dropout)
    _print_activations(fc1)

    # 全连接层2
    fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
    fc2 =tf.add(tf.matmul(fc2, weights['wd2']),biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    # dropout
    fc2=tf.nn.dropout(fc2,dropout)
    _print_activations(fc2)

    # 输出层
    out = tf.add(tf.matmul(fc2, weights['out']) ,biases['out'])
    _print_activations(out)
    return out

#构建模型，定义损失函数和优化器，并构建评估函数

# 构建模型
pred = alex_net(x, weights, biases, keep_prob)

# 定义损失函数和优化器
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)) 
#这里定义损失函数时调用tf.nn.softmax_cross_entropy_with_logits() 函数必须使用参数命名的方式来调用 (logits=pred, labels=y)不然会报错。
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 评估函数
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


#训练模型和评估模型

iterator = load_data('./datasets/train-00000-of-00001.tfrecord', \
                     image_shape=input_shape, \
                     batch_size=batch_size)
train_images, train_labels = iterator.get_next()


# 初始化变量
init = tf.global_variables_initializer()

# 开启一个训练
with tf.Session() as sess:
    sess.run(init)
    sess.run(iterator.initializer)
    step = 1
    # 开始训练，直到达到training_iters，即200000
    while step * batch_size < training_iters:
        #获取批量数据
        try:
            batch_x, batch_y = sess.run([train_images, train_labels])
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
            if step % display_step == 0:
                # 计算损失值和准确度，输出
                loss,acc = sess.run([cost,accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
                print ("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
            step += 1
        except tf.errors.OutOfRangeError: 
            sess.run(iterator.initializer)
    # print ("Optimization Finished!")
    # # 计算测试集的精确度
    # print ("Testing Accuracy:",
    #        sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
    #                                      y: mnist.test.labels[:256],
    #                                      keep_prob: 1.}))