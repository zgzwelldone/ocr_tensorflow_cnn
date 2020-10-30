import os

import tensorflow as tf
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

rows = 28
cols = 28

images_to_extract = 100

save_dir = "./tmp/result"

mnist = input_data.read_data_sets("./tmp/data/", one_hot=False)
session = tf.Session()

image_shape = session.run(tf.shape(mnist.train.images))
images_count = image_shape[0]
pixels_per_image = image_shape[1]

label_shape = session.run(tf.shape(mnist.train.labels))
labels_count = label_shape[0]
labels = mnist.train.labels

if images_count == labels_count:
    print("数据集中共包含%s张图片，和%s个标签" % (images_count, labels_count))
    print("每张图片包含%s个像素" % pixels_per_image)
    print("数据类型：%s" % mnist.train.images.dtype)

    if mnist.train.images.dtype == "float32":
        for i in range(0, images_to_extract):
            for n in range(pixels_per_image):
                if mnist.train.images[i][n] != 0:
                    mnist.train.images[i][n] = 255
            if ((i + 1) % 50) == 0:
                print("图像浮点数值扩展进度：已转换%s张，共需转换%s张" % (i + 1, images_to_extract))

    for i in range(10):
        save_path = "%s/%s/" % (save_dir, i)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    index = [0 for x in range(0, 10)]
    for i in range(0, images_to_extract):
        img = Image.new("L", (cols, rows))
        for m in range(rows):
            for n in range(cols):
                img.putpixel((n, m), int(mnist.train.images[i][n + m * cols]))
        digit = labels[i]
        path = "%s/%s/%s.png" % (save_dir, digit, index[digit])
        index[digit] += 1
        img.save(path)
