import cv2
import numpy as np

from torch.utils.data import DataLoader
#from HP_dataset import HP_dataset
from fiveJenre_Dataset import fiveJenre_Dataset

import torch
#from HandLSTM import LSTMmodel
from  LSTMmodel import LSTMmodel
import torch.optim as optim
import torch.nn as nn
import os
from Accuracy import calc_acc
#from create_confusion_matrix import calc_confusion_matrix
from fivejenre_create_confusion_matrix import calc_confusion_matrix

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
# train_dataset = HP_dataset(train_path_txt, os.path.join(filename, 'fivejenre_classes.txt'),seq, train_path_videos,(180,220) )# (180,220) is frame size for all frames

#train_path_videos = filename+"/train"
train_path_videos = os.path.join(filename, 'train_frames_120perIntervalsOfVideo')
train_dataset = fiveJenre_Dataset(train_path_videos, os.path.join(filename, 'fivejenre_classes.txt'),seq, (180,220) )# (180,220) is frame size for all frames


train_batch_size = 8 # if not work decrease to 4
#train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=0)

# test_path_videos = filename+"/test"
# test_path_txt = os.path.join(filename, 'test_frames_120perIntervalsOfVideo')
# test_dataset = HP_dataset(test_path_txt, os.path.join(filename, 'fivejenre_classes.txt'), seq, test_path_videos,(180,220))


test_path_videos = os.path.join(filename, 'test_frames_120perIntervalsOfVideo')
test_dataset = fiveJenre_Dataset(test_path_videos, os.path.join(filename, 'fivejenre_classes.txt'), seq, (180,220))
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

if __name__ == '__main__':

    #isBi = True # need to run one with tru and one with false
    isBi = True
    ####### LSTM Params #########
    #output_size = 16
    output_size = 5#delete problematic jenres which did not recognized by the model or confused to many other jenres
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

    # input: the parameters to be optimized
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss = nn.NLLLoss()

    # epochs and training
    old_acc = 0
    best_accuracy = 0
    epoch_num_of_best_acc = 0
    model_name_best_path= ""
    our_accuracy=0#current accuracy!!! may not the best one
    epochs = 35
    num_of_jenre_name = "NumOfJenre: "+str(output_size)+"_model: "
    tepoch= trange(epochs,desc='epoches Progress Bar', unit="epoch",position=0)
    for ep in range(epochs):
        with tqdm(train_loader, unit="batch",position=1,leave=False) as train_epoch:
            train_epoch.set_description(f"Train_Epoch {ep+1}")#cause ep start from o and end in epoches-1
            #print(f'epoch {ep + 1:< 4}', end='')
            #print("\n")
            total_loss = 0
            data_cnt = 0
            loss_out = 0
            #train_data_counter =0
            #print("start train\n")
            for hp_data, label in train_epoch:
                #train_data_counter=train_data_counter+1
                #print(train_data_counter)


                hp_data = hp_data.to(device)
                label = label.to(device)
                #print(hp_data.shape)
                output = net(hp_data)
                #loss_out = loss_out + loss(output, label)
                #print("loss begore change\n")
                #print(loss_out)
                loss_out =loss(output, label)
                #print("loss after change\n")
                #print(loss_out)
                loss_out.backward()
                optimizer.step()
                optimizer.zero_grad()
                #print("loss after optimizer calc\n")
                #print(loss_out)


                # if data_cnt % batch_size == 0 and data_cnt != 0:
                #     # print('Progress =',data_cnt//batch_size,'/',len(train_loader)//batch_size)
                #     # loss_out.backward: calculates the back-propagation algorithm
                #     loss_out.backward()
                #     # optimizes the model params
                #     optimizer.step()
                #     optimizer.zero_grad()
                #     loss_out = 0
                #
                # data_cnt = data_cnt + 1

            ####### Test model over valdiation test / test set
            #print("loss current training epoch\n")
            #print(loss_out)
            if((ep+1) % 5 ==0): #because 0<=0<=34, so we want last epoch to be count
                accurcy = 0
                with torch.no_grad():
                    net.eval()
                    #test_data_counter = 0
                    #print("start test\n")
                    with tqdm(test_loader, unit="batch",leave=False,position=2) as test_epoch:
                        test_epoch.set_description(f"Test_Epoch {ep+1}")#cause ep start from o and end in epoches-1
                        for hp_data, label in test_epoch:
                            #test_data_counter=test_data_counter+1
                            #print(test_data_counter)
                            hp_data = hp_data.to(device)
                            label = label.to(device)
                            output = net(hp_data)
                            if torch.argmax(output) == label:
                                accurcy = accurcy + 1

                    #print(f"accuracy={((accurcy / len(test_loader))*100)}")
                    our_accuracy = ((accurcy / len(test_loader)) * 100)
                    if(model_name_best_path == ""):
                        old_acc= 0
                    if(our_accuracy>best_accuracy):
                        if (old_acc != 0): #because first one with 0 acc , did not saved
                            curr_model_path = model_name_best_path+".pth"
                            os.remove(curr_model_path)#delete old path
                        best_accuracy=our_accuracy
                        epoch_num_of_best_acc = ep+1 #cause ep start from o and end in epoches-1
                        old_acc = our_accuracy
                        model_name_best_path = num_of_jenre_name+model_type + "_" + filename + "_isBi:_" + str(
                            isBi) + "_accuracy=" + str(best_accuracy)
                        torch.save(net.state_dict(), f'{model_name_best_path}.pth')#keep new path of best model

                    #add loss and accuracy to progress bar
                    tepoch.set_postfix(curr_accuracy=((accurcy / len(test_loader)) * 100),curr_loss=loss_out.item(),theBest_acc=best_accuracy,theEpochNumof_best_acc=epoch_num_of_best_acc)
                    tepoch.update(1)
                    net.train() #its only tells the model we are now training, so he knows to act differently where it needs
            else:
                tepoch.update(1)
    #our_accuracy = ((accurcy / len(test_loader))*100)
    #filename = model_type+"_"+filename +"_isBi:_"+str(isBi)+"_accuracy="+str(our_accuracy)
    #torch.save(net.state_dict(), f'{filename}.pth')
    if(best_accuracy == 0):
        print("\n\nproblem with model , best accuracy is 0\n")
    else:
        print("\n\nour best accuracy is: ")
        print(best_accuracy)
        print("best model saved, TODO: add currect calculation confusion matrix and etc., for result\n")

    print("\ncalc confusion matrix for model\n")
    calc_confusion_matrix(model_name_best_path, isBi=True)
    print("\nfinished all the code\n")
    #calc_acc(filename, isBi) #need  to fix inside the function, check borak if need, or there islibrary func to it
    #calc_confusion_matrix(filename, isBi) #need  to fix inside the function, check borak if need, or there islibrary func to it
