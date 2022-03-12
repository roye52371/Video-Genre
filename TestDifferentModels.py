import cv2
import numpy as np

from torch.utils.data import DataLoader

from Video_Dataset import fiveJenre_Dataset
from Video_Dataset import nineJenre_Dataset
from Video_Dataset import sixteenJenre_Dataset
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


def loadAndTestModel(Model_name,numOfJenre,classes_file): #model is str, numOfJenre is number, classes_file is txt file
    filename = 'Dataset70_30' # OR "Dataset70_30"( to run second time with 'Dataset80_20')

    init()

    print("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #hand_points = 42 * 3

    #seq should be according to frames created per video in offline proccesing in convertedVideosToFrames.ipynb
    seq=120#num of frames to take from one video, maybe to change , defend on what video we reading


    test_path_videos = os.path.join(filename, 'test_frames_120perIntervalsOfVideo')


    if(numOfJenre == 5):
        test_dataset = fiveJenre_Dataset(test_path_videos,os.path.join(filename, classes_file), seq,(180, 220))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)
    if (numOfJenre == 9):
        test_dataset = nineJenre_Dataset(test_path_videos, os.path.join(filename, classes_file), seq, (180, 220))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)
    if (numOfJenre == 16):
        test_dataset = sixteenJenre_Dataset(test_path_videos, os.path.join(filename, classes_file), seq, (180, 220))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)



    #isBi = True # need to run one with tru and one with false
    isBi = True
    ####### LSTM Params #########
    #output_size = 16
    output_size = numOfJenre #delete problematic jenres which did not recognized by the model or confused to many other jenres
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
    model_path = Model_name
    net.load_state_dict(torch.load(f'{model_path}.pth'))



    accurcy = 0
    net.eval()
    #test_data_counter = 0
    #print("start test\n")
    with tqdm(test_loader, unit="batch",leave=False,position=2) as test_epoch:
        test_epoch.set_description(f"Test_Model with {numOfJenre} Jenres")#cause ep start from o and end in epoches-1
        for hp_data, label in test_epoch:
            #test_data_counter=test_data_counter+1
            #print(str(curr_video_path[0]))
            hp_data = hp_data.to(device)
            label = label.to(device)
            output = net(hp_data)
            if torch.argmax(output) == label:
                accurcy = accurcy + 1

    our_accuracy = ((accurcy / len(test_loader)) * 100)
    print("\n")
    print("Model Name: ")
    print(Model_name)
    print("Num Of Jenre: ")
    print(str(numOfJenre))
    print("Model Accuracy: ")
    print(str(our_accuracy))
    print("\n")


if __name__ == '__main__':
    loadAndTestModel("NumOfJenre_5_model_CNN+LSTM_Dataset70_30_isBi_True_accuracy=54.90196078431373",5,'fivejenre_classes.txt')
    loadAndTestModel("CNN+LSTM_Dataset70_30_isBi_True_accuracy=42.37472766884531", 9,'ninejenre_classes.txt')
    loadAndTestModel("CNN+LSTM_Dataset70_30_isBi_True_accuracy=24.019607843137255", 16,'sixteenjenre_classes.txt')
