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

    # 随机生成字串，长度固定
    # 返回text,及对应的向量
    def random_text(self):
        text = ''
        vecs = np.zeros((self.idCard.max_size * self.idCard.len))
        size = self.idCard.max_size
        for i in range(size):
            c = random.choice(self.idCard.char_set)
            vec = self.char2vec(c)
            text = text + c
            vecs[i * self.idCard.len:(i + 1) * self.idCard.len] = np.copy(vec)
        return text, vecs

    # 根据生成的text，生成image,返回标签和图片元素数据
    def gen_image(self):
        text, vec = self.random_text()
        img = np.zeros([32, 256, 3])
        color_ = (255, 255, 255)  # white
        pos = (0, 0)
        text_size = 21
        image = self.image_util.draw_text(img, pos, text, text_size, color_)
        # 仅返回单通道值，颜色对于汉字识别没有什么意义
        return image[:, :, 2], text, vec

    # 单字转向量
    def char2vec(self, c):
        vec = np.zeros(self.idCard.len)
        for j in range(self.idCard.len):
            if self.idCard.char_set[j] == c:
                vec[j] = 1
        return vec

    # 向量转文本
    def vec2text(self, vecs):
        text = ''
        v_len = len(vecs)
        for i in range(v_len):
            if vecs[i] == 1:
                text = text + self.char_set[i % self.len]
        return text


if __name__ == '__main__':
    genObj = IdCardTrainDataUtil()
    image_data, label, vec = genObj.gen_image()
    cv2.imshow('image', image_data)
    cv2.waitKey(0)
