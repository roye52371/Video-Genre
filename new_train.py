import cv2
import numpy as np

from torch.utils.data import DataLoader
from HP_dataset import HP_dataset
import torch
from HandLSTM import LSTMmodel
import torch.optim as optim
import torch.nn as nn
import os
from Accuracy import calc_acc
from confusion_matrix import calc_confusion_matrix
#from EL.split_train_test import split
from colorama import init
from colorama import Fore, Back, Style

filename = 'Dataset70_30' # OR "Dataset70_30"( to run second time with 'Dataset80_20')


init()

print("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#hand_points = 42 * 3

seq=30#num of frames to take from one video
train_path = os.path.join(filename, 'train')
train_dataset = HP_dataset(train_path, os.path.join(filename, 'classes.txt'),seq,(180,220) )# (180,220) is frame size for all frames

batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

test_path = os.path.join(filename, 'test')
test_dataset = HP_dataset(test_path, os.path.join(filename, 'classes.txt'), seq,(180,220))
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

if __name__ == '__main__':
    """
    for isBi in [False, True]:
        for ratio in [60, 90, 70, 80]:
            if (not isBi and ratio != 90) or (isBi and ratio == 60):
                continue
            
            print(Back.RED, '=============')
            print(Style.RESET_ALL, end='')
            print(Fore.YELLOW, f'ratio: {ratio}, isBi: {isBi}')
            print(Back.RED, '=============')
            print(Style.RESET_ALL)
            split(ratio / 100)
            filename = f'weights_{ratio}{100 - ratio}_{isBi}'
            """
    isBi = True
    ####### LSTM Params #########
    output_size = 15
    # input of lstm
    #latent_dim = hand_points
    latent_dim = 512# roye and dekel latent dim according to borak
    # the vector length received from the past lstm run
    hidden_dim = 256
    # how many frames to look back (check it)
    #n_layers = 2
    n_layers = 1 # roye and dekel num of lstm layers according to borak
    net = LSTMmodel(output_size, latent_dim, hidden_dim, n_layers, model_name='Basic', isBi=isBi)
    # use the available device in the model
    net = net.to(device)

    # input: the parameters to be optimized
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss = nn.NLLLoss()

    # epochs and training
    epochs = 35
    for ep in range(epochs):
        print(f'epoch {ep + 1:< 4}', end='')
        total_loss = 0
        data_cnt = 0
        loss_out = 0
        for hp_data, label in train_loader:

            hp_data = hp_data.to(device)
            label = label.to(device)
            output = net(hp_data)
            loss_out = loss_out + loss(output, label)

            if data_cnt % batch_size == 0 and data_cnt != 0:
                # print('Progress =',data_cnt//batch_size,'/',len(train_loader)//batch_size)
                # loss_out.backward: calculates the back-propagation algorithm
                loss_out.backward()
                # optimizes the model params
                optimizer.step()
                optimizer.zero_grad()
                loss_out = 0

            data_cnt = data_cnt + 1

        ####### Test model over valdiation test / test set
        accurcy = 0
        with torch.no_grad():
            net.eval()
            for hp_data, label in test_loader:
                hp_data = hp_data.to(device)
                label = label.to(device)
                output = net(hp_data)
                if torch.argmax(output) == label:
                    accurcy = accurcy + 1
        print(f"accuracy={int((accurcy / len(test_loader))*100)}")
        net.train()
    torch.save(net.state_dict(), f'{filename}.pth')
    print("model saved, TODO: add currect calculation for result\n")
    calc_acc(filename, isBi)
    calc_confusion_matrix(filename, isBi)
