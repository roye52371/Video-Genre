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

# todo: create new HP_Dataset which contains: tensor = torch.zeros((len(data)=50(frames), self.hand_points=126)) as 2D tensor
"""
for id, frm in enumerate(data):
    tensor[id, :] = torch.Tensor(frm)
"""
class HP_dataset(Dataset):
    def __init__(self, train_videos_path, classes_path,seq_size, videos_path, resize_image=(180,220)):
        #self.train_videos_paths_txt = glob.glob(os.path.join(train_videos_path, '*', '*.txt'))
        self.train_videos_paths_txt = glob.glob(os.path.join(train_videos_path, '*','*','*'))# keep all frame video folder intervals paths
        #print(self.train_videos_paths_txt)
        #self.train_videos_paths_txt = self.train_videos_paths_txt[0:8] # delete this line
        #print(len(self.train_videos_paths_txt))
        #the line above is only for checking small number of data to check faster a full run
        #self.hand_points = hand_points
        #print(videos_path)
        self.videos_path = videos_path
        #print(self.videos_path)
        self.seq_size = seq_size #should be according to frames created per video in offline proccesing
        self.resize_image= resize_image
        random.shuffle(self.train_videos_paths_txt)

        with open(classes_path, "r") as read_file:
            self.class_num = json.load(read_file)# check if this is the way to read the classes numbers


        # this is the main method that we will use
        # getitem is used let's use with array calls

    def __getitem__(self, index):
        #print(self.videos_path)

        folder_video_path = self.train_videos_paths_txt[index]
        #print(folder_video_path)
        with open(folder_video_path,"r") as interval_frames:
            frames_array_index= interval_frames.read().splitlines()
        int_frames_array_index= [int(numeric_string) for numeric_string in frames_array_index]
        #print(int_frames_array_index)
        #frames_jenre = glob.glob(os.path.join(folder_video_path,'*.jpg'))
        #print(folder_video_path)
        #print(frames_jenre)
        #frames_jenre.sort(key=len)#to make sure frame_8 comes before framee 11 and etc
        #print(frames_jenre)

        # TODO: below reading video
        # with open(video_path, "r") as read_file: # need to change to reading a video, probably using opev cv videoCapture
        #    data = json.load(read_file)

        #tensor = torch.zeros((len(data), 1, self.hand_points))
        tensor = torch.zeros(self.seq_size, 3, self.resize_image[0], self.resize_image[1])
        path = os.path.normpath(folder_video_path)
        b = path.split(os.sep)
        curr_jenre = b[len(b) - 3]
        video_number = b[len(b) - 2]
        video_curr_path = self.videos_path+"/"+curr_jenre+"/"+video_number+".mp4"
        #print(video_curr_path)
        #for id, frm in enumerate(data):
        #    tensor[id,3,:, :] = torch.Tensor(frm)

        # video_frames = []
        int_frames_array_index.sort() # just for make sure the numers aree from small to big, if didnot take like that  from txt
        #print(int_frames_array_index)#check if sorted
        cap = cv2.VideoCapture(video_curr_path)
        if(cap.isOpened()):

            #     numberofframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # frames size in video
            #
            #     jumping_frames = int(np.floor(numberofframes / self.seq_size) ) # need to take frame after this number of times
            #     frame_index_array = []
            totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            count=0
            for i in range(0, self.seq_size):  # data size is the video size, check if start from 0 or 1 and end with size or size+1
                myFrameNumber = int_frames_array_index[i]


                # check for valid frame number
                if myFrameNumber >= 0 & myFrameNumber <= totalFrames-1:
                    # set frame position
                    #cap.set(cv2.CAP_PROP_POS_FRAMES, myFrameNumber)
                    while count!=myFrameNumber:
                        ret, frame = cap.read() # read the specific frame using setting its index in the video
                        count = count+1
                    #reading the accuratte frame, now count == myFrameNumber
                    ret, frame = cap.read()  # read the specific frame using setting its index in the video
                    count = count + 1 # finish reading now update to next one to try to read
                    #check if the frame was re
                    if ret == True:
                        resizearr = np.resize(frame, (3, 180, 220))
                        tensor[i] = torch.from_numpy(resizearr)

                    else:
                        print("problem reading frame in position(ret!=True): ")
                        print(myFrameNumber) #because counter updated before this printing to be myFrameNumber+1, so does not print here count
                        print("in the video: ")
                        print(folder_video_path)
                        print("while video size is: ")
                        print(totalFrames)
                        exit()

                else:
                    print("illegall frame index ask to be taken:\n")
                    print("frame number: ")
                    print(myFrameNumber)
                    print("while video size is: ")
                    print(totalFrames)
                    exit()

        else:
            print("cap could not open - in video:\n")
            print(folder_video_path)
            exit()
        cap.release()


        tensor = tensor/255;# normalize tensor
        #print(tensor.size())

        #path = os.path.normpath(folder_video_path)
        #b = path.split(os.sep)
        #print(b)
        #print(b[len(b)-3])
        # b[len(b)-3] is the jenre I think
        label = self.class_num[b[len(b) - 3]] #to check
        #label = self.class_num[os.path.dirname(video_path).split('\\')[-1]]
        # label = self.class_num[os.path.dirname(video_path).split('/')[-1]]
        label = torch.tensor(label)
        label = label.long()

        return tensor, label

    def __len__(self):  # return count of sample we have
        return len(self.train_videos_paths_txt)


# # todo: example of old main, maybe useful
# def prepor_json():
#     import json
#
#     # Opening JSON file
#     f = open('extracted_landmarks_test.json')
#
#     # returns JSON object as
#     # a dictionary
#     dataset_ = 'EL/test'
#     data = json.load(f)
#     width = 1980
#     height = 1080
#
#     class_dict = {}
#     for ky_num, ky in enumerate(data.keys()):
#         class_dict[ky] = ky_num
#
#         for ex_n, example in enumerate(data[ky]):
#
#             file_name_path = os.path.join(dataset_, ky, str(ex_n) + '.txt')
#             os.makedirs(os.path.join(dataset_, ky), exist_ok=True)
#             with open(file_name_path, "w") as write_file:
#
#                 dict_data = {}
#                 for n_frme, frme_hp in enumerate(example):
#                     # print('n_frme=',n_frme)
#                     vector_hp_One_Frame = frme_hp
#                     normlized = []
#                     for idx, el in enumerate(vector_hp_One_Frame):
#                         if idx % 2 == 0:
#                             normlized.append(vector_hp_One_Frame[idx] / width)
#                         else:
#                             normlized.append(vector_hp_One_Frame[idx] / height)
#
#                     dict_data[n_frme] = normlized
#                 json.dump(dict_data, write_file)
#
#         # Closing file
#         f.close()
#     print(class_dict)
#     with open(os.path.join('EL', 'classes.txt'), "w") as write_file:
#         json.dump(class_dict, write_file)

if __name__ == '__main__':
     filename = 'Dataset70_30'  # OR "Dataset70_30"( to run second time with 'Dataset80_20')
     train_path_videos = filename + "/train"

     print("cuda:0" if torch.cuda.is_available() else "cpu")
     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

     # hand_points = 42 * 3

     seq = 120  # num of frames to take from one video
     train_path = os.path.join(filename, 'train_frames_120perIntervalastxt')
     train_dataset = HP_dataset(train_path, os.path.join(filename, 'classes.txt'), seq, train_path_videos,
                                (180, 220))  # (180,220) is frame size for all frames

     #batch_size = 1
     train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)

     for hp_data, label in tqdm(train_loader,desc = 'tqdm() Progress Bar'):
         hp_data = hp_data.to(device)
         label = label.to(device)
         print(hp_data.shape)
         print(label)
         break
#
#      filename = 'Dataset70_30'  # OR "Dataset70_30"( to run second time with 'Dataset80_20')
#      model_type = "CNN+LSTM"
#      accurcy=4
#      test_loader = [1,2]
#      isBi=True
#      our_accuracy=6
#      accurcy = int((accurcy / len(test_loader)) * 100)
#      filename = model_type + "_" + filename + "_isBi:_" + str(isBi) + "_accuracy=" + str(our_accuracy)
#      print(filename)

#
# #     #### THIS APPLIED ONCE !!!!!
# #     # prepor_json()
# #     #################################
# #
# #     train_path = os.path.join('EL/train')
# #     train_dataset = DataLoader(train_path,'EL/classes.txt')
# #     for i in train_dataset:
# #      print(i[0].size(),i[1])
