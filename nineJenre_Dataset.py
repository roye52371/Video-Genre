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
class nineJenre_Dataset(Dataset):
    def __init__(self, videos_path, classes_path,seq_size, resize_image=(180,220)):
        #self.train_videos_paths_txt = glob.glob(os.path.join(train_videos_path, '*', '*.txt'))
        #self.videos_paths = glob.glob(os.path.join(videos_path, '*','*','*.mp4'))# keep all frame video folder intervals paths
        #self.videos_paths = self.videos_paths[0:8] # delete this line

        #start comment: what to do when wanting to NOT take Jenres from our DATASETS
        allvideosnotsureneeded = glob.glob(os.path.join(videos_path, '*','*','*.mp4'))# keep all frame video folder intervals paths
        self.videos_paths=[]
        print("size dataset before changes:\n")
        print(len(allvideosnotsureneeded))
        for vid in allvideosnotsureneeded:
            path = os.path.normpath(vid)
            b = path.split(os.sep)
            jenre= b[len(b)-3]
            if((jenre != "Animation") & (jenre != "Ice Hockey") & (jenre != "Judo") & (jenre != "Soccer")
                    & (jenre != "Swimming(in Pool)") & (jenre != "Tennis") & (jenre != "Volleyball")):
                #chosee jenres to take out
                #in if statement enter only if it is not one of the jenres you want out
                #afterwards, go to evert classes.txt files in your different datatsets
                #and delete it(Dataset, Dataset70_30,Dataset80_20)
                #than update in server, this(HP_dataset.py) file
                #and classes.txt files in all needed places in server code
                self.videos_paths.append(vid)
        print("size dataset after changes:\n")
        print(len(self.videos_paths))
        #print(self.videos_paths)


        #end start commet: what to do when wanting to NOT take Jenres from our DATASETS



        self.seq_size = seq_size #should be according to frames created per video in offline proccesing
        self.resize_image= resize_image
        random.shuffle(self.videos_paths)

        with open(classes_path, "r") as read_file:
            self.class_num = json.load(read_file)# check if this is the way to read the classes numbers

        print("jenre size:\n")
        print(len(self.class_num))
        # this is the main method that we will use
        # getitem is used let's use with array calls

    def __getitem__(self, index):
        #print(self.videos_path)

        video_path = self.videos_paths[index]
        #print(video_path)
        #tensor = torch.zeros((len(data), 1, self.hand_points))
        tensor = torch.zeros(self.seq_size, 3, self.resize_image[0], self.resize_image[1])

        cap = cv2.VideoCapture(video_path)
        #if(cap.isOpened()):
        totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        #print(totalFrames)
        frame_counter = 0
        if (cap.isOpened()):
            if(totalFrames == self.seq_size):
                for i in range(0, self.seq_size):  # data size is the video size, check if start from 0 or 1 and end with size or size+1
                    ret, frame = cap.read() # read the specific frame using setting its index in the video
                    #check if the frame was re
                    if ret == True:
                        resizearr = np.resize(frame, (3, 180, 220))
                        tensor[i] = torch.from_numpy(resizearr)
                        #frame_counter = frame_counter+1

                    else:
                        print("problem reading frame in position(ret!=True): ")
                        print(i) #because counter updated before this printing to be myFrameNumber+1, so does not print here count
                        print("in the video: ")
                        print(video_path)
                        print("while video size is: ")
                        print(totalFrames)
                        exit()

            else:
                print("illegall interval size video\n")
                print("in the video: ")
                print(video_path)
                print("our wanted seq size is: ")
                print(self.seq_size)
                print("while video size is: ")
                print(totalFrames)
                exit()

        else:
            print("cap could not open - in video:\n")
            print(video_path)
            exit()
        cap.release()

        # if(frame_counter != self.seq_size):
        #     print("reading less than seq_size framesillegall interval size video\n")
        #     print("in the video: ")
        #     print(video_path)
        #     print("our wanted seq size is: ")
        #     print(self.seq_size)
        #     print("while readed frames were: ")
        #     print(frame_counter)
        #     exit()



        tensor = tensor/255;# normalize tensor
        #print(tensor.size())

        path = os.path.normpath(video_path)
        b = path.split(os.sep)
        #print(b)
        jenre= b[len(b)-3]
        #print(jenre)
        # b[len(b)-3] is the jenre I think
        label = self.class_num[jenre] #to check
        #print(label)
        #label = self.class_num[os.path.dirname(video_path).split('\\')[-1]]
        # label = self.class_num[os.path.dirname(video_path).split('/')[-1]]
        label = torch.tensor(label)
        label = label.long()

        return tensor, label

    def __len__(self):  # return count of sample we have
        return len(self.videos_paths)


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

# if __name__ == '__main__':
#      filename = 'Dataset70_30'  # OR "Dataset70_30"( to run second time with 'Dataset80_20')
#      #train_path_videos = filename + "/train"
#
#      print("cuda:0" if torch.cuda.is_available() else "cpu")
#      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#      # hand_points = 42 * 3
#
#      seq = 120  # num of frames to take from one video
#      train_path = os.path.join(filename, 'train_frames_120perIntervalsOfVideo')
#      train_dataset = HP_dataset(train_path, os.path.join(filename, 'classes.txt'), seq,
#                                 (180, 220))  # (180,220) is frame size for all frames
#
#      #batch_size = 1
#      train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
#
#      for hp_data, label in tqdm(train_loader,desc = 'tqdm() Progress Bar'):
#          hp_data = hp_data.to(device)
#          label = label.to(device)
#          print(hp_data.shape)
#          print(label)
#          break
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
