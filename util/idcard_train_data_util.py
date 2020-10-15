"""
身份证训练数据生成工具类

@author: zhaogzh
"""
import random

import cv2
import numpy as np
from bean.idcard_info import IdCard
from util.image_util import ImageUtil


class IdCardTrainDataUtil(object):

    def __init__(self):
        self.idCard = IdCard()
        self.image_util = ImageUtil('.\\fonts\\OCR-B.ttf')

    # 随机生成定长(self.idCard.max_size:18)字串
    # 返回text,及对应的向量
    def gen_random_text(self):
        text = ''
        zero_vectors = np.zeros((self.idCard.max_size * self.idCard.length))
        size = self.idCard.max_size
        for i in range(size):
            char = random.choice(self.idCard.char_set)
            vector = self.char2vec(char)
            text = text + char
            zero_vectors[i * self.idCard.length:(i + 1) * self.idCard.length] = np.copy(vector)
        return text, zero_vectors

    # 根据生成的text，生成image,返回标签和图片元素数据
    def gen_train_image(self):
        text, text_vectors = self.gen_random_text()
        img = np.zeros([32, 256, 3])
        color_ = (255, 255, 255)  # white
        pos = (0, 0)
        text_size = 21
        image = self.image_util.draw_text(img, pos, text, text_size, color_)
        # 仅返回单通道值，颜色对于汉字识别没有什么意义
        return image[:, :, 2], text, text_vectors

    # 单字转向量
    def char2vec(self, input_char):
        zero_vectors = np.zeros(self.idCard.length)
        for i in range(self.idCard.length):
            if self.idCard.char_set[i] == input_char:
                zero_vectors[i] = 1
        return zero_vectors

    # 向量转文本
    def vec2text(self, input_vectors):
        text = ''
        v_len = len(input_vectors)
        for i in range(v_len):
            if input_vectors[i] == 1:
                text = text + self.idCard.char_set[i % self.idCard.length]
        return text


if __name__ == '__main__':
    genObj = IdCardTrainDataUtil()
    image_data, label, vectors = genObj.gen_train_image()
    cv2.imshow('image', image_data)
    cv2.waitKey(0)
