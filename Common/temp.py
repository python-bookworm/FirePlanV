# -*- coding: utf-8 -*-
import os
import glob
import numpy as np

# def delfile(path):
#     file_names=glob.glob(path+r"*")
#     for filename in file_names:
#         try:
#             os.remove(filename)
#         except:
#             try:
#                 os.rmdir(filename)
#             except:
#                 delfile(filename)
#                 os.rmdir(filename)
path = "../Output/train.txt"


# delfile(path)

# with open(path) as f:
#     lines = f.readlines()
# np.random.shuffle(lines)
#
# num_val = int(len(lines))
# print(num_val)

def get_path(path):
    temp_file = ""
    for files in os.listdir(path):
        temp_file = files
    print(temp_file)
    temp_name=temp_file.split(".")[-1]

    # spl=path.split(".")
    # outdir=spl[0]+"_result"+"."+spl[-1]
    # print(outdir)
    # print(spl[0])
    pass
get_path(r"C:\Users\hanjx\Desktop\项目整理\深度学习项目\点滴液杂质\image")