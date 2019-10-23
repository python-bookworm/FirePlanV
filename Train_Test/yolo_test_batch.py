# -*- coding: utf-8 -*-
"""
批量测试图像
测试的结果存放在路径outdir = "VOC2007/SegmentationClass
"""
import argparse
import os
from Train_Test.yolo import YOLO
from PIL import Image


import glob
def detect_img(yolo):
    path = "VOC2007/Images/*.jpg"
    outdir = "VOC2007/SegmentationClass"
    for jpgfile in glob.glob(path):
        img = Image.open(jpgfile)
        img = yolo.detect_image(img)
        img.save(os.path.join(outdir, os.path.basename(jpgfile)))
    yolo.close_session()
FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + 'logs/best_weights.h5'
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + 'Model_data/tiny_yolo_anchors.txt'
    )

    parser.add_argument(
        '--classes.txt', type=str,
        help='path to class definitions, default ' + 'Model_data/voc_classes.txt'
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + "1"
    )

    # parser.add_argument(
    #     '--image', default=False, action="store_true",
    #     help='Image detection mode, will ignore all positional arguments'
    # )
    '''
    Command line positional arguments -- for video detection mode
    '''
    # parser.add_argument(
    #     "--input", nargs='?', type=str,required=False,default='./path2your_video',
    #     help = "Video input path"
    # )

    # parser.add_argument(
    #     "--output", nargs='?', type=str, default="",
    #     help = "[Optional] Video output path"
    # )
    FLAGS = parser.parse_args()

    print("type",type(FLAGS))
    print("FLAGS",FLAGS)

    detect_img(YOLO(**vars(FLAGS)))

#     if FLAGS.image:
#         """
#         Image detection mode, disregard any remaining command line arguments
#         """
#         #print("Image detection mode")
#         if "input" in FLAGS:
#              print("error")
# #            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
# #        detect_img(YOLO(**vars(FLAGS)))
#     elif "input" in FLAGS:
# #        print("error")
#          #print("Image detection mode")
#          #print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
#          detect_img(YOLO(**vars(FLAGS)))
# #        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
# #    else:
# #        print("Must specify at least video_input_path.  See usage with --help.")
#     #detect_img(yolo)