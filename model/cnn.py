import tensorflow as tf

from tf_cnn_lstm_ctc import OUTPUT_SHAPE


# 创建模型权重张量
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.5)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 创建一个卷积层
def conv2d(x, weight, stride=(1, 1), padding='SAME'):
    return tf.nn.conv2d(x, weight, strides=[1, stride[0], stride[1], 1], padding=padding)


# 创建一个最大池化层
def max_pool(x, ksize=(2, 2), stride=(2, 2)):
    return tf.nn.max_pool(x, ksize=[1, ksize[0], ksize[1], 1], strides=[1, stride[0], stride[1], 1], padding='SAME')


# 创建一个平均池化层
def avg_pool(x, ksize=(2, 2), stride=(2, 2)):
    return tf.nn.avg_pool(x, ksize=[1, ksize[0], ksize[1], 1], strides=[1, stride[0], stride[1], 1], padding='SAME')


# 定义CNN网络，处理图片，
def generate_convolution_network():
    # 输入数据
    inputs = tf.placeholder(tf.float32, [None, None, OUTPUT_SHAPE[0]])

    # 第一组卷积层, 32*256*1 => 16*128*48
    # 卷积核:5*5,卷积核个数：48
    weight_conv_group1 = weight_variable([5, 5, 1, 48])
    bias_conv_group1 = bias_variable([48])
    x_expanded = tf.expand_dims(inputs, 3)
    # 卷积层, 32*256*1 => 32*256*48
    activation_function_conv_group1 = tf.nn.relu(conv2d(x_expanded, weight_conv_group1) + bias_conv_group1)
    # 池化层, 32*256*48 => 16*128*48
    pool_group1 = max_pool(activation_function_conv_group1, ksize=(2, 2), stride=(2, 2))

    # 第二组, 16*128*48 => 16*64*64
    # 卷积核:5*5*48,卷积核个数：64
    weight_conv_group2 = weight_variable([5, 5, 48, 64])
    bias_conv_group2 = bias_variable([64])
    # 卷积层, 16*128*48  => 16*128*64
    activation_function_conv_group2 = tf.nn.relu(conv2d(pool_group1, weight_conv_group2) + bias_conv_group2)
    # 池化层, 16*128*64 => 16*64*64
    pool_group2 = max_pool(activation_function_conv_group2, ksize=(2, 1), stride=(2, 1))

    # 第三层, 16*64*64 => 8*32*128
    # 卷积核:5*5*64,卷积核个数：128
    weight_conv_group3 = weight_variable([5, 5, 64, 128])
    bias_conv_group3 = bias_variable([128])
    # 卷积层, 16*64*64  => 16*64*128
    activation_function_conv_group3 = tf.nn.relu(conv2d(pool_group2, weight_conv_group3) + bias_conv_group3)
    # 池化层, 16*64*128 => 8*32*128
    pool_group3 = max_pool(activation_function_conv_group3, ksize=(2, 2), stride=(2, 2))

    # 全连接,神经元个数为256
    weight_full_connect = weight_variable([16 * 8 * OUTPUT_SHAPE[1], OUTPUT_SHAPE[1]])
    bias_full_connect = bias_variable([OUTPUT_SHAPE[1]])
    conv_layer_flat = tf.reshape(pool_group3, [-1, 16 * 8 * OUTPUT_SHAPE[1]])
    features = tf.nn.relu(tf.matmul(conv_layer_flat, weight_full_connect) + bias_full_connect)
    shape = tf.shape(features)
    features = tf.reshape(features, [shape[0], OUTPUT_SHAPE[1], 1])
    return inputs, features
