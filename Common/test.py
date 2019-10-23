# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 12:59:20 2018

@author: Administrator
"""

import os
import random


class A(object):
    def __init__(self):
        self.trainval_percent = 0.1
        self.train_percent = 0.9

    def get_paths(self, path1, path2):
        self.xmlfilepath = path1
        self.txtsavepath = path2

    def fun(self):
        # xmlfilepath = './VOC2007/Annotations'
        # txtsavepath = './VOC2007/ImageSets/Main'
        total_xml = os.listdir(self.xmlfilepath)
        num = len(total_xml)
        list = range(num)
        tv = int(num * self.trainval_percent)
        tr = int(tv * self.train_percent)

        trainval = random.sample(list, tv)
        train = random.sample(trainval, tr)

        ftrainval = open('././Output/trainval.txt', 'w')
        ftest = open('././Output/test.txt', 'w')
        ftrain = open('././Output/train.txt', 'w')
        fval = open('././Output/val.txt', 'w')

        for i in list:
            name = total_xml[i][:-4] + '\n'
            if i in trainval:
                ftrainval.write(name)
                if i in train:
                    ftest.write(name)
                else:
                    fval.write(name)
            else:
                ftrain.write(name)

        ftrainval.close()
        ftrain.close()
        fval.close()
        ftest.close()
