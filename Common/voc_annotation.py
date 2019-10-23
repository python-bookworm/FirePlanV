import xml.etree.ElementTree as ET
from os import getcwd, listdir

sets = ['train', 'val', 'test']


# classes_path = 'Model_data/voc_classes.txt'

class Voc_Anno(object):
    def __init__(self):
        # self.classes_path = 'Model_data/voc_classes.txt'
        self.class1 = ""

    def get_class_path(self, class_path, pics_path, xml_path):
        self.classes_path = class_path
        self.pics_path = pics_path
        self.xml_path = xml_path

    def get_classes(self, classes_path):
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def get_classname(self):
        class_names = self.get_classes(self.classes_path)
        # classes.txt = ["LouJin"]
        self.class1 = class_names

    # print("22222",self.class1)

    def convert_annotation(self, image_id, list_file):
        # 解析xml文件
        in_file = open(self.xml_path + "/%s.xml" % (image_id), encoding='UTF-8')
        tree = ET.parse(in_file)
        root = tree.getroot()

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            # 缺陷类型
            cls = obj.find('name').text
            # print("333",cls)
            # print("4444",self.class1)
            if cls not in self.class1 or int(difficult) == 1:
                continue
            cls_id = self.class1.index(cls)
            # 缺陷位置
            xmlbox = obj.find('bndbox')
            b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
                 int(xmlbox.find('ymax').text))
            list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
            # print("tests:::", b)

    def do_fun(self):
        wd = getcwd()
        # print("test_path", wd)
        temp_file = ""
        for files in listdir(self.pics_path):
            temp_file = files
        #print(temp_file)
        temp_name = temp_file.split(".")[-1]

        for image_set in sets:
            image_ids = open('./Output/%s.txt' % (image_set)).read().strip().split()
            # 当前文件夹下的txt文件
            list_file = open("./TrainTxt/" + '%s.txt' % (image_set), 'w')

            for image_id in image_ids:
                # list_file.write('%s/VOC%s/JPEGImages/%s.jpg' % (wd, year, image_id))
                # list_file.write(self.pics_path+"/%s.bmp"%(image_id))
                list_file.write(self.pics_path + "/%s.%s" % (image_id, temp_name))
                self.convert_annotation(image_id, list_file)
                list_file.write('\n')
            list_file.close()
        #print("训练文件生成结束.....")
