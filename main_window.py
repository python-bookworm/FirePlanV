# -*- coding: utf-8 -*-
import Main_window
from Main_window.visualization_ui import *
from PyQt5.QtCore import QCoreApplication, QThread, pyqtSignal,QMutex
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QMessageBox, QFileDialog
from PyQt5.QtGui import QIcon
# import cv2
import sys
import os
import time
QtCore.QCoreApplication.addLibraryPath(
    "C:\\Users\\hanjx\\AppData\\Local\\Programs\\Python\\Python35\\Lib\\site-packages\\PyQt5\\Qt\\plugins\\platforms")

sys.setrecursionlimit(1000000)
from Train_Test.yolo_test import Recongnit
from Train_Test.tiny_train import _main,setting_gpu
from Common.test import A
from Common.voc_annotation import Voc_Anno
from threading import Thread
from multiprocessing import Process, Pool

qmut_1=QMutex()
qmut_2=QMutex()
qmut_3=QMutex()
class EmittingStream(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))


# 线程1继承QThread,处理模型训练操作
class Runthread(QThread):
    #  通过类成员对象定义信号对象
    _signal = pyqtSignal()

    def __init__(self):
        super(Runthread, self).__init__()

    def get_path(self, path2, path3, pics_path, xml_path,gpu_num,batch_size,epoch):
        # self.input_dir_path = path1
        self.save_dir_path = path2
        self.classes_path = path3
        self.pics_path = pics_path
        self.xml_path = xml_path
        self.gpu_num=gpu_num
        self.batch_size=batch_size
        self.epoch=epoch
    def __del__(self):
        self.wait()

    def run(self):
        qmut_1.lock()
        self._signal.emit()
        P = A()
        P.get_paths(self.pics_path, self.xml_path)
        P.fun()

        V = Voc_Anno()
        V.get_class_path(self.classes_path, self.pics_path, self.xml_path)
        V.get_classname()
        V.do_fun()

        print("训练模型.....")
        setting_gpu(self.gpu_num)
        _main(self.save_dir_path, self.classes_path,self.batch_size,self.epoch)
        qmut_1.unlock()

# 线程2
class Runthread2(QThread):

    #  通过类成员对象定义信号对象
    _signal = pyqtSignal()
    def __init__(self):
        super(Runthread2, self).__init__()

    def get_path(self, class_path, pic_path, xml_path):
        self.class_path = class_path
        self.pic_path = pic_path
        self.xml_path = xml_path
        pass

    def __del__(self):
        self.wait()

    def run(self):
        qmut_2.lock()
        #print("正在生成训练文件.....")
        self._signal.emit()
        P = A()
        P.get_paths(self.pic_path, self.xml_path)
        P.fun()
        app = Voc_Anno()
        app.get_class_path(self.class_path, self.pic_path, self.xml_path)
        app.get_classname()
        app.do_fun()
        # print("生成训练文件.....")
        qmut_2.unlock()

# 线程3处理图片识别操作
class Runthread3(QThread):
    #  通过类成员对象定义信号对象
    _signal = pyqtSignal()

    def __init__(self,sigle_pic_path, batchs_pics_path, model_path, classes_path):
        super(Runthread3,self).__init__()
        self.pic_path=sigle_pic_path
        self.bach_pics_path=batchs_pics_path
        self.weights_path=model_path
        self.class_path=classes_path

    def get_paths(self, single_pic_path, bach_path, weights_path, voc_classes_path):
        self.pic_path = single_pic_path
        self.bach_pics_path = bach_path
        self.weights_path = weights_path
        self.class_path = voc_classes_path
        pass

    # def __del__(self):
    #     self.wait()

    def run(self):
        qmut_3.lock()
        _signal = pyqtSignal()
        R = Recongnit()
        R.get_paths(self.pic_path, self.bach_pics_path,  self.weights_path, self.class_path)
        R.args()
        qmut_3.unlock()
        pass


# class UI_SecondWindow():
#     def __init__(self):
#         super().__init__()
#         self.resize(200,200)
#         self.move(200,200)

def Batches(sigle_pic_path, batchs_pics_path, model_path, classes_path):
    batchs = Recongnit()
    batchs.get_paths(sigle_pic_path, batchs_pics_path, model_path, classes_path)
    batchs.args()
    pass

class MyWindow(QtWidgets.QMainWindow,Ui_MainWindowv):
    def __init__(self):
        super(MyWindow, self).__init__()

        self.setupUi(self)
        # self.set_second_ui(self)

        sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)
        sys.stder = EmittingStream(textWritten=self.normalOutputWritten)

        # ---------------------------------界面初始化——-------------------------------#
        #设置窗口
        self.setWindowTitle("缺陷检测")
        #self.setWindowIcon(QIcon("././Settings/logo.png"))
        # 系统参数初始化
        self.sys_parameter_initialization()
        # 加载训练图片
        self.load_train_path.clicked.connect(self.load_train_pics)

        # 加载标注文件
        self.load_xml_path.clicked.connect(self.load_xml_file)

        # 输入图片路径设置
        self.pb_data_input_path.clicked.connect(self.set_data_input_path)

        # 输入文件路径设置
        # self.pb2_data_input_path.clicked.connect(self.set_dir_input_path)

        # 输出保存文件路径
        self.pb_save_dir_path.clicked.connect(self.set_save_dir_path)

        # 重新设置参数
        self.pb_refresh.clicked.connect(self.refresh_parameters)

        # 生成训练文件
        # self.make_train_file.clicked.connect(self.make_file)

        # 退出界面设置
        self.pb_quit.clicked.connect(self.set_quit)

        self.thread = None
        # self.thread2 = None
        self.thread3 = None

        # 运行训练
        self.pb_run.clicked.connect(self._run)

        # 识别单张图片
        self.recognit_pic.clicked.connect(self.recognite_pic_fun)
        # 批量识别图片
        self.recognit_batch.clicked.connect(self.recognit_batch_fun)
        # 刷新
        self.refresh2.clicked.connect(self.refresh2_fun)

        # 加载模型
        self.load_model.clicked.connect(self.load_model_fun)

        # 识别图片
        self.recognition.clicked.connect(self.recongnition_fun)

        # 加载类别文件
        self.load_class.clicked.connect(self.load_class_fun)
        # 设置参数
        # self.set_parameters.clicked.connect(self.set_parmers)

    # 加载训练图片
    def load_train_pics(self):
        self.load_train_pic_path.clear()
        load_dir_path = QFileDialog.getExistingDirectory(self, "输入文件路径")
        if not os.path.exists(load_dir_path):
            QMessageBox.warning(self, "提示", "输入文件目录不存在，请重新选择")
            return
        self.load_train_images = load_dir_path
        self.load_train_pic_path.setText(self.load_train_images)
        print("加载训练数据:", self.load_train_images)

    # 加载xml文件路径
    def load_xml_file(self):
        self.load_xml_path_2.clear()
        load_xml_file_path = QFileDialog.getExistingDirectory(self, "输入文件路径")
        if not os.path.exists(load_xml_file_path):
            QMessageBox.warning(self, "提示", "输入文件目录不存在，请重新选择")
            return
        self.xml_file_path = load_xml_file_path
        self.load_xml_path_2.setText(self.xml_file_path)
        print("加载标注文件:", self.xml_file_path)

    # 设置输入图片路径
    def set_data_input_path(self):
        # input_data_path = QFileDialog.getExistingDirectory(self, "输入图片路径", self.data_input_path)
        # if not os.path.exists(input_data_path):
        #     QMessageBox.warning(self, "提示", "输入文件目录不存在，请重新选择")
        #     return
        self.le_data_input_path.clear()
        input_dir_path, dir_type = QFileDialog.getOpenFileNames(self, "选择文件", "datasets")
        # print("11111",input_dir_path)
        # 判断文件合法性

        if len(input_dir_path):
            for each_path in input_dir_path:
                if ".txt" not in each_path or not os.path.exists(each_path):
                    QMessageBox.warning(self, "提示", "文件选择错误，请重新选择！")
                    return
            self.input_data_path1 = input_dir_path[0]
            self.le_data_input_path.setText(self.input_data_path1)
            # text_content=self.le_data_input_path.text()
            # print(text_content)

            print("输入类型文件:", self.input_data_path1)
        else:
            return

    # 生成训练文件
    def make_file(self):
        class_path = self.le_data_input_path.text()
        pics_path = self.load_train_pic_path.text()
        xml_path = self.load_xml_path_2.text()

        self.thread2 = Runthread2()
        # self.thread.get_path()
        self.thread2.get_path(class_path, pics_path, xml_path)

        self.thread2._signal.connect(self.test)
        self.thread2.start()

    # 设置输入文件路径
    def set_dir_input_path(self):
        # input_dir_path = QFileDialog.getExistingDirectory(self, "输入文件路径", self.data_input_path)
        input_dir_path, dir_type = QFileDialog.getOpenFileNames(self, "选择文件", "datasets")
        # print("11111",input_dir_path)
        # 判断文件合法性
        if len(input_dir_path):

            for each_path in input_dir_path:
                if ".txt" not in each_path or not os.path.exists(each_path):
                    QMessageBox.warning(self, "提示", "文件选择错误，请重新选择！")
                    return

            self.input_data_path2 = input_dir_path[0]
            self.le2_data_input_path.setText(self.input_data_path2)
            print("输入训练文件:", self.input_data_path2)
        else:
            return

    # 设置输出文件保存路径
    def set_save_dir_path(self):
        self.le_save_dir_path.clear()
        save_dir_path = QFileDialog.getExistingDirectory(self, "输入文件路径", self.output_save_path)
        if not os.path.exists(save_dir_path):
            QMessageBox.warning(self, "提示", "输入文件目录不存在，请重新选择")
            return
        self.output_save_path = save_dir_path
        self.le_save_dir_path.setText(self.output_save_path)
        print("输出文件保存路径:", self.output_save_path)

    # 参数重新设置
    def refresh_parameters(self):
        self.te_screen_display.clear()
        # self.le2_data_input_path.clear()
        self.sys_parameter_initialization()

    # 退出窗口设置
    def set_quit(self):
        QCoreApplication.instance().quit()

    # 运行训练
    def _run(self):
        # input_dir_path = self.le2_data_input_path.text()
        save_dir_path = self.le_save_dir_path.text()
        classes_names_path = self.le_data_input_path.text()
        pics_path = self.load_train_pic_path.text()
        xml_path = self.load_xml_path_2.text()
        batch_size=self.comboBox_2.currentText()
        epoch=self.comboBox.currentText()
        # if not pics_path:
        #     QMessageBox.warning(self, "提示", "请加载训练集！")
        #     return
        # if not xml_path:
        #     QMessageBox.warning(self, "提示", "请加载标注文件！")
        #     return
        # if not classes_names_path:
        #     QMessageBox.warning(self, "提示", "请选择类别文件！")
        #     return
        # if not save_dir_path:
        #     QMessageBox.warning(self, "提示", "请选择文件保存路径！")
        #     return
        if self.checkBox_gpu.isChecked():
            gpu_num=1
        else:
            gpu_num=0

        self.thread = Runthread()
        self.thread.get_path(save_dir_path, classes_names_path, pics_path, xml_path,gpu_num,batch_size,epoch)
        self.thread._signal.connect(self.test)
        self.thread.start()

    # 加载单张图片
    def recognite_pic_fun(self):

        self.recognit_pic_path.clear()

        recog_pic_path, dir_type = QFileDialog.getOpenFileNames(self, "选择文件", "datasets")
        # print("11111",input_dir_path)
        # 判断文件合法性
        if len(recog_pic_path):
            for each_path in recog_pic_path:
                temp=each_path.split(".")[-1]
                if temp not in ["jpg","bmp","JPG","jpeg","BMP"] and not os.path.exists(each_path):
                    QMessageBox.warning(self, "提示", "文件选择错误，请重新选择！")
                    return

                # for i in [".jpg",".bmp", "JPG", ".jpeg", ".BMP"]:
                #     if i in each_path and os.path.exists(each_path):
                #         QMessageBox.warning(self, "提示", "文件选择错误，请重新选择！")
                #         return
                self.rec_pic_path = recog_pic_path[0]
                self.recognit_pic_path.setText(self.rec_pic_path)
                print("识别文件:", self.rec_pic_path)
                return
        else:
            return
        pass

    # 加载批量识别文件
    def recognit_batch_fun(self):
        self.recognit_batch_path.clear()
        temp_recongnit_path = QFileDialog.getExistingDirectory(self, "输入文件路径", self.output_save_path)
        if not os.path.exists(temp_recongnit_path):
            QMessageBox.warning(self, "提示", "输入文件目录不存在，请重新选择")
            return
        self.recongnit_batchs_path = temp_recongnit_path
        self.recognit_batch_path.setText(self.recongnit_batchs_path)
        print("图片文件:", self.recongnit_batchs_path)

    # 加载识别模型
    def load_model_fun(self):

        self.load_model_path.clear()
        load_model_path, model_type = QFileDialog.getOpenFileNames(self, "选择文件", "datasets")

        if len(load_model_path):

            for each_path in load_model_path:
                if ".h5" not in each_path or not os.path.exists(each_path):
                    QMessageBox.warning(self, "提示", "文件选择错误，请重新选择！")
                    return

            self.model_path = load_model_path[0]
            self.load_model_path.setText(self.model_path)
            print("加载模型文件:", self.model_path)
            return

        else:
            return
        pass

    # 加载类别文件
    def load_class_fun(self):

        self.load_class_path.clear()

        classes_path, dir_type = QFileDialog.getOpenFileNames(self, "选择文件", "datasets")
        # print("11111",input_dir_path)
        # 判断文件合法性
        if len(classes_path):
            for each_path in classes_path:
                if ".txt" not in each_path or not os.path.exists(each_path):
                    QMessageBox.warning(self, "提示", "文件选择错误，请重新选择！")
                    return

            self.class_path = classes_path[0]
            self.load_class_path.setText(self.class_path)
            print("输入训练文件:", self.class_path)
            return
        else:
            return
        pass

    # 刷新识别区域
    def refresh2_fun(self):
        self.recognit_pic_path.clear()
        self.recognit_batch_path.clear()
        self.load_model_path.clear()
        self.load_class_path.clear()
        self.label_img_show.clear()
        pass

    # 识别图片
    def recongnition_fun(self):
        self.label_img_show.clear()
        model_path = self.load_model_path.text()
        sigle_pic_path = self.recognit_pic_path.text()
        batchs_pics_path = self.recognit_batch_path.text()
        classes_path = self.load_class_path.text()
        # self.label_img_show.clear()

        # 判断加载文件的合法性
        if not model_path:
            QMessageBox.warning(self, "提示", "请加载模型！")
            return
        elif not classes_path:
            QMessageBox.warning(self, "提示", "请选择类型！")
            return
        elif not sigle_pic_path and not batchs_pics_path:
            QMessageBox.warning(self, "提示", "请选择识别图片！")
            return
        else:
            pass

        # 识别单张图并显示
        if len(sigle_pic_path):
            Sigle = Recongnit()
            Sigle.get_paths(sigle_pic_path, batchs_pics_path, model_path, classes_path)
            (img, outdir) = Sigle.args_test2()
            # outdir=sigle_pic_path
            image = QtGui.QPixmap(outdir).scaled(self.label_img_show.width(), self.label_img_show.height())
            self.label_img_show.setPixmap(image)
            self.Show_image(outdir)
            return

        if len(batchs_pics_path):
            # T=Thread(target=Batches,args=(sigle_pic_path, batchs_pics_path, model_path, classes_path))
            # T.start()
            batchs=Recongnit()
            batchs.get_paths(sigle_pic_path, batchs_pics_path, model_path, classes_path)
            batchs.args()
            # time.sleep(10)

            #self.thread3 = Runthread3(sigle_pic_path, batchs_pics_path, model_path, classes_path)
            # self.thread3.get_paths(sigle_pic_path, batchs_pics_path, model_path, classes_path)
            #self.thread3._signal.connect(self.test())
            #self.thread3.start()

            #return


        # def set_parmers(self):
        #     UI_Wind=UI_SecondWindow()
        #     UI_Wind.show()
        #     UI_Wind.exec_()
        pass

    def Show_image(self, outdir):

        pass

    #
    def test(self):
        # print("开始测试....")
        # print(i)
        pass

    # 参数初始化
    def sys_parameter_initialization(self):
        print("系统初始化为空.........")
        self.input_data_path1 = ''
        self.output_save_path = ""
        self.load_train_images = ""
        self.xml_file_path = ""
        self.rec_pic_path = ""
        self.recongnit_batchs_path = ""
        self.model_path = ""
        self.class_path = ''
        # 设置输入文件路径
        self.load_train_pic_path.setText(self.load_train_images)
        self.load_xml_path_2.setText(self.xml_file_path)
        self.le_data_input_path.setText(self.input_data_path1)
        self.recognit_pic_path.setText(self.rec_pic_path)
        self.recognit_batch_path.setText(self.recongnit_batchs_path)
        self.load_model_path.setText(self.model_path)
        self.load_class_path.setText(self.class_path)

        # 设置数据保存路径
        self.le_save_dir_path.setText(self.output_save_path)
        #self.label_size = self.label.size()

    def normalOutputWritten(self, text):
        """Append text to the QTextEdit."""
        # Maybe QTextEdit.append() works as well, but this is how I do it:
        cursor = self.te_screen_display.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.te_screen_display.setTextCursor(cursor)
        self.te_screen_display.ensureCursorVisible()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = MyWindow()
    w.show()
    sys.exit(app.exec_())
