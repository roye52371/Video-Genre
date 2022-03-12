import cv2
import numpy as np

from torch.utils.data import DataLoader
from video_dataset_with_path_name import vid_dataset_with_path_name
import torch
#from HandLSTM import LSTMmodel
from  LSTMmodel import LSTMmodel
import torch.optim as optim
import torch.nn as nn
import os
from Accuracy import calc_acc
from create_confusion_matrix import calc_confusion_matrix
#from EL.split_train_test import split
from colorama import init
from colorama import Fore, Back, Style
from tqdm import tqdm
from tqdm import trange

filename = 'Dataset70_30' # OR "Dataset70_30"( to run second time with 'Dataset80_20')
model_type = "CNN+LSTM"

init()

print("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#hand_points = 42 * 3

#seq should be according to frames created per video in offline proccesing in convertedVideosToFrames.ipynb
seq=120#num of frames to take from one video
# train_path_videos = filename+"/train"
# train_path_txt = os.path.join(filename, 'train_frames_120perIntervalsOfVideo')
# train_dataset = HP_dataset(train_path_txt, os.path.join(filename, 'classes.txt'),seq, train_path_videos,(180,220) )# (180,220) is frame size for all frames


# test_path_videos = filename+"/test"
# test_path_txt = os.path.join(filename, 'test_frames_120perIntervalsOfVideo')
# test_dataset = HP_dataset(test_path_txt, os.path.join(filename, 'classes.txt'), seq, test_path_videos,(180,220))


test_path_videos = os.path.join(filename, 'test_frames_120perIntervalsOfVideo')
test_dataset = vid_dataset_with_path_name(test_path_videos, os.path.join(filename, 'classes.txt'), seq, (180,220))
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

if __name__ == '__main__':

    #isBi = True # need to run one with tru and one with false
    isBi = True
    ####### LSTM Params #########
    #output_size = 16
    output_size = 9 #delete problematic jenres which did not recognized by the model or confused to many other jenres
    # input of lstm
    #latent_dim = hand_points
    latent_dim = 512# roye and dekel latent dim according to borak
    # the vector length received from the past lstm run
    hidden_dim = 256
    # how many frames to look back (check it)
    #n_layers = 2
    #n_layers = 1 # roye and dekel num of lstm layers according to borak
    n_layers=1
    net = LSTMmodel(output_size, latent_dim, hidden_dim, n_layers, model_name='Basic', isBi=isBi)
    # use the available device in the model
    net = net.to(device)

    #load model
    net = net.to(device)
    _use_new_zipfile_serialization = False
    model_path = "CNN+LSTM_Dataset70_30_isBi_True_accuracy=42.37472766884531"
    net.load_state_dict(torch.load(f'{model_path}.pth'))


    #first writing
    f = open("failed_toDetect_Videos.txt", "a")  # so we can append to the same file accuracy of different models
    f.write("model name_" + model_path+ "\n")
    f.close()

    accurcy = 0
    num_of_failed_sub_videos=0
    net.eval()
    #test_data_counter = 0
    #print("start test\n")
    with tqdm(test_loader, unit="batch",leave=False,position=2) as test_epoch:
        test_epoch.set_description(f"Test_Model")#cause ep start from o and end in epoches-1
        for hp_data, label,curr_video_path in test_epoch:
            #test_data_counter=test_data_counter+1
            #print(str(curr_video_path[0]))
            hp_data = hp_data.to(device)
            label = label.to(device)
            output = net(hp_data)
            if torch.argmax(output) == label:
                accurcy = accurcy + 1
            else:
                num_of_failed_sub_videos = num_of_failed_sub_videos+1
                f = open("failed_toDetect_Videos.txt","a")  # so we can append to the same file accuracy of different models
                f.write(str(curr_video_path) + "\n")
                f.close()


    f = open("failed_toDetect_Videos.txt","a")  # so we can append to the same file accuracy of different models
    f.write("num of videos failed to detected their currect jenre" + str(num_of_failed_sub_videos) +"\n")
    f.close()