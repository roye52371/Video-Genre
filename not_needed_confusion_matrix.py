from torch.utils.data import DataLoader
from HP_dataset import HP_dataset
import torch
from HandLSTM import LSTMmodel
import torch.optim as optim
import torch.nn as nn
import os, re
import numpy as np
import matplotlib.pyplot as plt
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
    plt.figure(figsize=(50, 50))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the Model')
    fig.colorbar(cax)
    fig.figure.tight_layout()
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'{filename}_martix.jpeg')
    plt.show()

    # calculating the: precision recall f1-score support
    prec_rcall = classification_report(lbl_lst, pred_lst, target_names=classnames)
    prec_rcall = prec_rcall.splitlines()
    with open(f'{filename}.txt', 'w+') as res:
        res.write('\n'.join(prec_rcall))
    for l in prec_rcall:
        if "accuracy" in l:
            print(Fore.GREEN, l)
            break


def calc_confusion_matrix(filename='weights', isBi=False):
    #############
    # Model
    #############
    print("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ####### LSTM Params #########
    output_size = 15
    # input of lstm
    latent_dim = 42 * 3

    # the vector length received from the past lstm run
    hidden_dim = 256
    # how many frames to look back (check it)
    n_layers = 2

    net = LSTMmodel(output_size, latent_dim, hidden_dim, n_layers, model_name='Basic', isBi=isBi)
    # use the available device in the model
    net = net.to(device)
    net.load_state_dict(torch.load(f'{filename}.pth'))
    net.eval()

    #############
    # DATASET
    #############
    test_path = os.path.join('EL', 'test')
    classes_path = os.path.join('EL', 'classes.txt')
    test_dataset = HP_dataset(test_path, classes_path, hand_points=latent_dim)
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
    for hp_data, label in test_loader:
        hp_data = hp_data.to(device)
        label = label.to(device)
        output = net(hp_data).cpu().detach().numpy()
        # print("output=", np.argmax(output))
        lbl_lst.append(inv_map[label.cpu().detach().numpy()[0]])
        pred_lst.append(inv_map[np.argmax(output)])

    classnames = list(class_num.keys())
    plot_m(lbl_lst, pred_lst, classnames, filename)


if __name__ == '__main__':
    calc_confusion_matrix(filename='weights_6040_True', isBi=True)
