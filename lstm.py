import tensorflow as tf

from tf_cnn_lstm_ctc import OUTPUT_SHAPE
from util.idcard_train_data_util import IdCardTrainDataUtil

# 定义一些常量
num_hidden = 64
num_layers = 1

obj = IdCardTrainDataUtil()
num_classes = obj.idCard.length + 1 + 1  # 10位数字 + blank + ctc blank


def generate_lstm_network():
    inputs = tf.placeholder(tf.float32, [None, None, OUTPUT_SHAPE[0]])

    # 定义ctc_loss需要的稀疏矩阵
    targets = tf.sparse_placeholder(tf.int32)

    # 1维向量 序列长度 [batch_size,]
    seq_len = tf.placeholder(tf.int32, [None])

    # 定义LSTM网络
    cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    outputs, _ = tf.nn.dynamic_rnn(cell, inputs, seq_len, dtype=tf.float32)

    shape = tf.shape(inputs)
    batch_s, max_timesteps = shape[0], shape[1]

    outputs = tf.reshape(outputs, [-1, num_hidden])
    weight = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1), name="weight")
    bias = tf.Variable(tf.constant(0., shape=[num_classes]), name="bias")
    logits = tf.matmul(outputs, weight) + bias
    logits = tf.reshape(logits, [batch_s, -1, num_classes])
    logits = tf.transpose(logits, (1, 0, 2))

    return logits, inputs, targets, seq_len