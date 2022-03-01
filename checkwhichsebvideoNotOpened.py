
import os
import math
import cv2
from pathlib import Path
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import os
import glob
import json
import numpy as np
import pandas as pd
import random
from PIL import Image, ImageDraw
import cv2
import torchvision as tv
import matplotlib.pyplot as plt
import torchvision
from functools import reduce
import random
from tqdm import tqdm

#check what not opended in test
filename = 'Dataset70_30' # OR "Dataset70_30"( to run second time with 'Dataset80_20')

test_path_videos = os.path.join(filename, 'test_frames_120perIntervalsOfVideo')
sub_videos_paths = glob.glob(os.path.join(test_path_videos, '*','*','*.mp4'))
#print(sub_videos_paths)
print(len(sub_videos_paths))
count_problematic_vid = 0
problematic_jenre= []
for our_video in sub_videos_paths:
    #print(our_video)
    cap = cv2.VideoCapture(our_video)
    #print("aaa\n")
    # print(video.isOpened())
    # framerate = video.get(5)
    # os.makedirs(dataset_path_frames_jenre_dir + "/video_" + str(int(count)))
    # filenamenotype= Path(file).stem
    # print(filenamenotype)
    # os.makedirs(dataset_path_frames_jenre_dir + "/video_" + filenamenotype)

    if (cap.isOpened()):
        pass #everything is ok



    else:
        print("cap could not open - in video:\n")
        print(our_video)
        path = os.path.normpath(our_video)
        b = path.split(os.sep)
        # print(b)
        jenre = b[len(b) - 3]
        problematic_jenre.append(jenre)
        count_problematic_vid = count_problematic_vid + 1
        #exit()
    cap.release()
    # print('done')


print("num of problematic videos not open with cap in test: ")
print(count_problematic_vid)
print("'problematic jenre\n")
print(problematic_jenre)
