# -*- coding: utf-8 -*-

import os
from Train_Test.yolo import YOLO
from PIL import Image
import glob


class Recongnit(object):
    def __init__(self):
        self.FLAGS = None
        self.yolo=YOLO()
    def get_paths(self, single_pic_path, bach_path, weights_path, voc_classes_path):
        self.pic_path = single_pic_path
        self.bach_pics_path = bach_path
        self.weights_path = weights_path
        self.class_path = voc_classes_path
        return


    def detect_single_pic(self):

        img = Image.open(self.pic_path)
        #yolo1 = YOLO()
        self.yolo.get_path(self.weights_path, self.class_path)
        img = self.yolo.detect_image(img)
        # 保存图片
        spl = self.pic_path.split(".")
        outdir = spl[0] + "_result" + "." + spl[-1]
        img.save(outdir)
        # yolo1.close_session()
        return (img, outdir)

    def detect_img(self):
        path = self.bach_pics_path + "/*.jpg"
        # outdir = "VOC2007/SegmentationClass"
        outdir = self.bach_pics_path + "/Result"
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        #yolo2 = YOLO()
        print(self.weights_path)
        self.yolo.get_path(self.weights_path, self.class_path)
        for jpgfile in glob.glob(path):
            #print("imgfile",jpgfile)
            img = Image.open(jpgfile)
            img = self.yolo.detect_image(img)
            img.save(os.path.join(outdir, os.path.basename(jpgfile)))
        self.yolo.close_session()
        return

    def args_test2(self):

        (img, outdir) = self.detect_single_pic()
        return (img, outdir)

    def args(self):
        self.detect_img()
        return
# if __name__ == '__main__':
#     app = Recongnit()
#     app.args()
