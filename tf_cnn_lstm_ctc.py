"""
tf CNN+LSTM+CTC 训练识别不定长数字字符图片

@author: zhaogzh
"""
import os
import time

from bean.constants import *
from model.lstm import generate_lstm_network
from util.idcard_train_data_util import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


def decode_sparse_tensor(sparse_tensor):
    decoded_indexes = list()
    current_i = 0
    current_seq = []
    for offset, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(offset)
    decoded_indexes.append(current_seq)
    result = []
    if len(sparse_tensor[1]) - 1 != decoded_indexes[len(decoded_indexes) - 1][
        len(decoded_indexes[len(decoded_indexes) - 1]) - 1]:
        print(len(sparse_tensor[1]) - 1, decoded_indexes[len(decoded_indexes) - 1])
    else:
        for index in decoded_indexes:
            result.append(decode_a_seq(index, sparse_tensor))
    return result


def decode_a_seq(indexes, spars_tensor):
    decoded = []
    for m in indexes:
        if spars_tensor[1][m] > 9:
            print("Index Error:%s" % spars_tensor[1][m])
        temp_string = DIGITS[spars_tensor[1][m]]
        decoded.append(temp_string)
    return decoded


def report_accuracy(decoded_list, test_targets):
    original_list = decode_sparse_tensor(test_targets)
    detected_list = decode_sparse_tensor(decoded_list)
    true_number = 0

    if len(original_list) != len(detected_list):
        print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
              " test and detect length desn't match")
        return
    print("T/F: original(length) <-------> detectcted(length)")
    for idx, number in enumerate(original_list):
        detect_number = detected_list[idx]
        hit = (number == detect_number)
        # print(hit, number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")")
        if hit:
            true_number = true_number + 1
    print("Test Accuracy:", true_number * 1.0 / len(original_list))


def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Create a sparse representention of x.
    Args:
        :param sequences:
        :param dtype:
    Returns:
        A tuple with (indices, values, shape)
        indices: 1152(64*18)*2, [[ 0  0], [ 0  1], [ 0  2], [ 0  3], [ 0  4], [ 0  5], [ 0  6], [ 0  7], [ 0  8], [ 0  9], [ 0 10], [ 0 11], [ 0 12], [ 0 13], [ 0 14], [ 0 15], [ 0 16], [ 0 17], [ 1  0], [ 1  1], [ 1  2], [ 1  3], [ 1  4], [ 1  5], [ 1  6], [ 1  7], [ 1  8], [ 1  9], ...
                 图片index, 字符index
        values: 1152, [0 5 1 4 8 5 2 3 3 2 7 4 5 6 9 1 6 0 7 1 3 1 2 7 6 0 8 0 9 4 9 9 9 7 6 0 1, 5 9 1 0 2 6 2 4 5 5 3 2 9 9 4 9 5 0 6 3 3 1 6 4 9 4 1 1 1 3 3 9 0 1 6 8 0, 0 8 8 7 1 2 8 9 7 7 6 7 6 4 6 8 6 6 9 5 7 0 8 1 7 5]
                字符标记
        shape: [64 18]
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)
    # 维度：1152*2,
    # [[ 0  0], [ 0  1], [ 0  2], [ 0  3], [ 0  4], [ 0  5], [ 0  6], [ 0  7], [ 0  8], [ 0  9], [ 0 10], [ 0 11], [ 0 12], [ 0 13], [ 0 14], [ 0 15], [ 0 16], [ 0 17], [ 1  0], [ 1  1], [ 1  2], [ 1  3], [ 1  4], [ 1  5], [ 1  6], [ 1  7], [ 1  8], [ 1  9], ...
    indices = np.asarray(indices, dtype=np.int64)
    # 维度：1152
    # [0 5 1 4 8 5 2 3 3 2 7 4 5 6 9 1 6 0 7 1 3 1 2 7 6 0 8 0 9 4 9 9 9 7 6 0 1, 5 9 1 0 2 6 2 4 5 5 3 2 9 9 4 9 5 0 6 3 3 1 6 4 9 4 1 1 1 3 3 9 0 1 6 8 0, 0 8 8 7 1 2 8 9 7 7 6 7 6 4 6 8 6 6 9 5 7 0 8 1 7 5]
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def get_next_batch(batch_size=128):
    """
    生成一个批次的训练图片数据
    Args:
        :param batch_size: 批次图片数量，默认128
    Returns:
        A tuple with (inputs, sparse_targets, seq_len)
        inputs: 64*256*32 64张训练图片数据
        sparse_targets: 参照sparse_tuple_from方法的返回值
        seq_len:
    """
    obj = IdCardTrainDataUtil()
    # (batch_size,256,32)
    inputs = np.zeros([batch_size, OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]])
    codes = []

    for i in range(batch_size):
        # 生成不定长度的字串
        image, text, vec = obj.gen_train_image()
        # np.transpose 矩阵转置 (32*256,) => (32,256) => (256,32)
        # 维度：64*256*32，64张生成图片
        inputs[i, :] = np.transpose(image.reshape((OUTPUT_SHAPE[0], OUTPUT_SHAPE[1])))
        codes.append(list(text))
    targets = [np.asarray(i) for i in codes]
    sparse_targets = sparse_tuple_from(targets)
    seq_len = np.ones(inputs.shape[0]) * OUTPUT_SHAPE[1]

    return inputs, sparse_targets, seq_len


def train():
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                               global_step,
                                               DECAY_STEPS,
                                               LEARNING_RATE_DECAY_FACTOR,
                                               staircase=True)
    logits, inputs, seq_len = generate_lstm_network()

    # 定义ctc_loss需要的稀疏矩阵
    targets = tf.sparse_placeholder(tf.int32)

    loss = tf.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=seq_len)
    cost = tf.reduce_mean(loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
    acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

    init = tf.global_variables_initializer()

    def do_batch():
        train_inputs, train_targets, train_seq_len = get_next_batch(BATCH_SIZE)
        feed = {inputs: train_inputs, targets: train_targets, seq_len: train_seq_len}

        train_loss, train_targets, train_logits, train_seq_len, train_cost, train_steps, _ = session.run(
            [loss, targets, logits, seq_len, cost, global_step, optimizer], feed)

        if train_steps > 0 and train_steps % REPORT_STEPS == 0:
            test_inputs, test_targets, test_seq_len = get_next_batch(BATCH_SIZE)
            test_feed = {inputs: test_inputs, targets: test_targets, seq_len: test_seq_len}
            test_decode, test_log_prob, test_accuracy = session.run([decoded[0], log_prob, acc], test_feed)
            report_accuracy(test_decode, test_targets)
        return train_cost, train_steps

    with tf.Session() as session:
        session.run(init)
        for curr_epoch in range(num_epochs):
            print("Epoch.......", curr_epoch)
            train_cost_sum = train_ler_sum = 0
            for batch in range(BATCHES):
                start = time.time()
                c, steps = do_batch()
                train_cost_sum += c * BATCH_SIZE
                seconds = time.time() - start
                print("Step:", steps, ", batch seconds:", seconds)

            train_cost_sum /= TRAIN_SIZE

            # 模型验证
            verify_inputs, verify_targets, verify_seq_len = get_next_batch(BATCH_SIZE)
            val_feed = {inputs: verify_inputs, targets: verify_targets, seq_len: verify_seq_len}

            val_cost, val_ler, lr, steps = session.run([cost, acc, learning_rate, global_step], feed_dict=val_feed)

            log = "Epoch {}/{}, steps = {}, 训练损失值 = {:.3f}, train_ler = {:.3f}, 验证损失值 = {:.3f}, 验证准确度 = {:.3f}, time = {:.3f}s, 学习率 = {}"
            print(log.format(curr_epoch + 1, num_epochs, steps, train_cost_sum, train_ler_sum, val_cost, val_ler,
                             time.time() - start, lr))


if __name__ == '__main__':
    train()
