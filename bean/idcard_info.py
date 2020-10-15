"""
身份证信息

@author: zhaogzh
"""


class IdCard(object):
    def __init__(self):
        # 身份证包含的数字集合
        self.number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        # 身份证包含的字符集合
        self.char_set = self.number
        # 身份证字符集长度
        self.len = len(self.char_set)
        # 训练数据字符长度
        self.max_size = 18
