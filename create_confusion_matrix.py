from torch.utils.data import DataLoader
from tqdm import tqdm

from HP_dataset import HP_dataset
import torch
#from HandLSTM import LSTMmodel
from  LSTMmodel import LSTMmodel
import torch.optim as optim
import torch.nn as nn
import os, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import json
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from colorama import init
from colorama import Fore, Back, Style

init()


def plot_m(lbl_lst, pred_lst, classnames, filename):
    labels = classnames
    print('labels=', labels)
    cm = confusion_matrix(lbl_lst, pred_lst, labels, normalize='true')
    print("\n")
    print(cm)
    print("\n")
    plt.figure(figsize=(50, 50))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the Model')
    fig.colorbar(cax)
    #fig.figure.tight_layout() # did plt.tight_layout() on current figure instead
    plt.tight_layout()
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'{filename}_martix.jpeg',bbox_inches= 'tight')

    #plt.show()

    # calculating the: precision recall f1-score support
    prec_rcall = classification_report(lbl_lst, pred_lst, target_names=classnames)
    prec_rcall = prec_rcall.splitlines()
    with open(f'{filename}.txt', 'w+') as res:
        res.write('\n'.join(prec_rcall))
    for l in prec_rcall:
        if "accuracy" in l:
            print(Fore.GREEN, l)
            break


def calc_confusion_matrix(filename, isBi):
    #############
    # Model
    #############
    print("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ####### LSTM Params #########
    isBi = True
    ####### LSTM Params #########
    output_size = 9
    # input of lstm
    # latent_dim = hand_points
    latent_dim = 512  # roye and dekel latent dim according to borak
    # the vector length received from the past lstm run
    hidden_dim = 256
    # how many frames to look back (check it)
    # n_layers = 2
    # n_layers = 1 # roye and dekel num of lstm layers according to borak
    n_layers = 1
    net = LSTMmodel(output_size, latent_dim, hidden_dim, n_layers, model_name='Basic', isBi=isBi)
    # use the available device in the model
    net = net.to(device)

    # input: the parameters to be optimized
    #optimizer = optim.Adam(net.parameters(), lr=0.001)
    #loss = nn.NLLLoss()
    # use the available device in the model
    net = net.to(device)
    _use_new_zipfile_serialization = False
    net.load_state_dict(torch.load(f'{filename}.pth'))
    #net.load_state_dict(torch.load(f'{filename}.pth'), strict=False)
    net.eval()

    #############
    # DATASET
    #############
    dataset_name = 'Dataset70_30'  # OR "Dataset70_30"( to run second time with 'Dataset80_20')
    # seq should be according to frames created per video in offline proccesing in convertedVideosToFrames.ipynb
    seq = 120  # num of frames to take from one video

    classes_path = os.path.join(dataset_name, 'classes.txt')
    test_path_videos = os.path.join(dataset_name, 'test_frames_120perIntervalsOfVideo')
    test_dataset = HP_dataset(test_path_videos, os.path.join(dataset_name, 'classes.txt'), seq, (180, 220))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

    ##############3
    # Classses
    #############

    with open(classes_path, "r") as read_file:
        class_num = json.load(read_file)
    inv_map = {v: k for k, v in class_num.items()}

    index = 1
    total_loss = 0
    data_cnt = 0
    loss_out = 0
    lbl_lst = []
    pred_lst = []
    with tqdm(test_loader, unit="batch", leave=False, position=2) as test_epoch:
        test_epoch.set_description(f"Test for confusion matrix")  # cause ep start from o and end in epoches-1
        for hp_data, label in test_epoch:
            hp_data = hp_data.to(device)
            label = label.to(device)
            #print(hp_data.size())
            output = net(hp_data).cpu().detach().numpy()
            # print("output=", np.argmax(output))
            lbl_lst.append(inv_map[label.cpu().detach().numpy()[0]])
            pred_lst.append(inv_map[np.argmax(output)])

    classnames = list(class_num.keys())
    plot_m(lbl_lst, pred_lst, classnames, filename)


if __name__ == '__main__':
    calc_confusion_matrix(filename='CNN+LSTM_Dataset70_30_isBi:_True_accuracy=24.019607843137255', isBi=True)
