from torchvision import transforms
from torch.utils.data import Dataset
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

# todo: create new HP_Dataset which contains: tensor = torch.zeros((len(data)=50(frames), self.hand_points=126)) as 2D tensor
"""
for id, frm in enumerate(data):
    tensor[id, :] = torch.Tensor(frm)
"""
class HP_dataset(Dataset):
    def __init__(self, train_videos_path, classes_path,seq_size=30, resize_image=(180,220)):
        #self.train_videos_paths = glob.glob(os.path.join(train_videos_path, '*', '*.txt'))
        self.train_videos_paths = glob.glob(os.path.join(train_videos_path, '*', '*.mp4'))
        #self.hand_points = hand_points
        self.seq_size = seq_size
        self.resize_image= resize_image
        random.shuffle(self.train_videos_paths)

        with open(classes_path, "r") as read_file:
            self.class_num = json.load(read_file)# check if this is the way to read the classes numbers


        # this is the main method that we will use
        # getitem is used let's use with array calls

    def __getitem__(self, index):
        #label = None
        video_path = self.train_videos_paths[index]
        # TODO: below reading video
        # with open(video_path, "r") as read_file: # need to change to reading a video, probably using opev cv videoCapture
        #    data = json.load(read_file)

        #tensor = torch.zeros((len(data), 1, self.hand_points))
        tensor = torch.zeros(self.seq, 3, self.resize_image[0], self.resize_image[1])
        #for id, frm in enumerate(data):
        #    tensor[id,3,:, :] = torch.Tensor(frm)



        # extracting the video frames
        video_frames = []
        cap = cv2.VideoCapture(video_path)
        if(cap.isOpened()):

            numberofframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # frames size in video

            jumping_frames = np.floor(numberofframes / self.seq_size)  # need to take frame after this number of times
            frame_index_array = []
            for i in range(0, self.seq_size):  # data size is the video size, check if start from 0 or 1 and end with size or size+1
                # need to add index of frame  that devide with out reminder in self.seq_size from the specific video
                cap.set(cv2.CAP_PROP_POS_FRAMES,i*jumping_frames)
                ret, frame = cap.read()
                if ret == True:
                    resizearr = np.resize(frame, (3, 180, 220))
                    tensor[i] = torch.from_numpy(resizearr)
                else:
                    print("frame didnt extracted well or finished if shows uup try to delete this printing\n")
                    break

        else:
            print("cap could not open\n")
            exit()
        cap.release()





        # extracting keeping

        #keeping the indexes of frames we want to take
        """
        jumping_frames = len(video_frames)/ self.seq_size # need to take frame after this number of times
        frame_index_array=[]
        for i in range(0, len(video_frames)): # data size is the video size, check if start from 0 or 1 and end with size or size+1
            #need to add index of frame  that devide with out reminder in self.seq_size from the specific video
            if(i%jumping_frames==0):
                frame_index_array.append(i)

        """

        for idx, id_frm in enumerate(frame_index_array):
            #TODO: below reading specif frame from the video(video(id_frm)), and resize it to (3,180, 220)(using pytorch or pytorch probably)
            resizearr = np.resize(video_frames(id_frm), (3, 180, 220))
            tensor[idx] = torch.from_numpy(resizearr)

            #tensor[idx,3,:,:] = cv2.resize(video_frames(id_frm),(3,180,220))# check if need 3 or juct 180,220


        # maybe we needd to change to hotencoder
        # label = torch.zeros(1, len(self.class_num.keys()), dtype=torch.long)
        # label[0, self.class_num[os.path.dirname(video_path).split('/')[-1]]] = 1
        #reading jenre name and take it value- aka its label number
        label = self.class_num[os.path.dirname(video_path).split('\\')[-1]]
        # label = self.class_num[os.path.dirname(video_path).split('/')[-1]]
        label = torch.tensor(label)
        label = label.long()

        return tensor, label

    def __len__(self):  # return count of sample we have
        return len(self.train_videos_paths)


# todo: example of old main, maybe useful
def prepor_json():
    import json

    # Opening JSON file
    f = open('extracted_landmarks_test.json')

    # returns JSON object as
    # a dictionary
    dataset_ = 'EL/test'
    data = json.load(f)
    width = 1980
    height = 1080

    class_dict = {}
    for ky_num, ky in enumerate(data.keys()):
        class_dict[ky] = ky_num

        for ex_n, example in enumerate(data[ky]):

            file_name_path = os.path.join(dataset_, ky, str(ex_n) + '.txt')
            os.makedirs(os.path.join(dataset_, ky), exist_ok=True)
            with open(file_name_path, "w") as write_file:

                dict_data = {}
                for n_frme, frme_hp in enumerate(example):
                    # print('n_frme=',n_frme)
                    vector_hp_One_Frame = frme_hp
                    normlized = []
                    for idx, el in enumerate(vector_hp_One_Frame):
                        if idx % 2 == 0:
                            normlized.append(vector_hp_One_Frame[idx] / width)
                        else:
                            normlized.append(vector_hp_One_Frame[idx] / height)

                    dict_data[n_frme] = normlized
                json.dump(dict_data, write_file)

        # Closing file
        f.close()
    print(class_dict)
    with open(os.path.join('EL', 'classes.txt'), "w") as write_file:
        json.dump(class_dict, write_file)
# if __name__ == '__main__':
#
#     #### THIS APPLIED ONCE !!!!!
#     # prepor_json()
#     #################################
#
#     train_path = os.path.join('EL/train')
#     train_dataset = DataLoader(train_path,'EL/classes.txt')
#     for i in train_dataset:
#      print(i[0].size(),i[1])
