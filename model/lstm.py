import tensorflow as tf

from bean.constants import OUTPUT_SHAPE
from util.idcard_train_data_util import IdCardTrainDataUtil

# 定义一些常量
num_hidden = 64
num_layers = 1

obj = IdCardTrainDataUtil()
num_classes = obj.idCard.length + 1  # 10位数字 + ctc blank


def generate_lstm_network():
    inputs = tf.placeholder(tf.float32, [None, None, OUTPUT_SHAPE[0]])

    seq_len = tf.placeholder(tf.int32, [None])

    # 定义LSTM网络
    cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)

    # 获取输入数据的纬度
    shape = tf.shape(inputs)
    batch_s, max_timesteps = shape[0], shape[1]

    outputs = tf.reshape(outputs, [-1, num_hidden])
    weight = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1), name="weight")
    bias = tf.Variable(tf.constant(0., shape=[num_classes]), name="bias")
    logits = tf.matmul(outputs, weight) + bias
    logits = tf.reshape(logits, [batch_s, -1, num_classes])
    logits = tf.transpose(logits, (1, 0, 2))

    return logits, inputs, seq_len
