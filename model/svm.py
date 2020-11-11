import numpy as np
import tensorflow as tf
from sklearn import datasets

# 获取数据
iris = datasets.load_iris()
x_values = np.array([[x[0], x[3]] for x in iris.data])
y_values = np.array([1 if y == 0 else -1 for y in iris.target])

# 分离训练和测试集
train_indices = np.random.choice(len(x_values), int(len(x_values) * 0.8), replace=False)
test_indices = np.array(list(set(range(len(x_values))) - set(train_indices)))
x_values_train = x_values[train_indices]
x_values_test = x_values[test_indices]
y_values_train = y_values[train_indices]
y_values_test = y_values[test_indices]

batch_size = 100

# 初始化feeding
x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# 创建权值参数
A = tf.Variable(tf.random_normal(shape=[2, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# 定义线性模型: y = Ax + b
model_output = tf.subtract(tf.matmul(x_data, A), b)

# Declare vector L2 'norm' function squared
l2_norm = tf.reduce_sum(tf.square(A))

# Loss = max(0, 1-pred*actual) + alpha * L2_norm(A)^2
alpha = tf.constant([0.01])
classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model_output, y_target))))
loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))

my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

# 持久化
saver = tf.train.Saver()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training loop
    for i in range(20000):
        rand_index = np.random.choice(len(x_values_train), size=batch_size)
        rand_x = x_values_train[rand_index]
        rand_y = np.transpose([y_values_train[rand_index]])
        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    saver.save(sess, "./result/svm.ckpt")
