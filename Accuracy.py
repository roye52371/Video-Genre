from torch.utils.data import DataLoader
from HP_dataset import HP_dataset
import torch
from HandLSTM import LSTMmodel
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns


def calc_acc(filename="weights", isBi=False):
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

    classes_path = os.path.join('EL', 'classes.txt')
    #############
    # DATASET
    #############
    test_path = os.path.join('EL', 'test')
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
    pred_lst = [0] * output_size
    counts = [0] * output_size
    for hp_data, label in test_loader:
        hp_data = hp_data.to(device)
        label = label.to(device)
        output = net(hp_data).cpu().detach().numpy()
        # print("output=", np.argmax(output))
        counts[label.cpu().detach().numpy()[0]] += 1
        if label.cpu().detach().numpy()[0] == np.argmax(output):
            pred_lst[np.argmax(output)] += 1
    for i, count in enumerate(counts):
        pred_lst[i] = pred_lst[i] / count

    sns.set_theme(style="whitegrid")
    w = [[x, y] for x, y in zip(list(inv_map.values()), pred_lst)]
    data = sorted(w, key=lambda x: x[1])
    fig = sns.barplot(y=[lab[0] for lab in data], x=[lab[1] for lab in data])
    fig.figure.tight_layout()
    plt.title("Accuracy of the model")
    plt.savefig(f'{filename}_accuracy.jpeg')
    plt.show()


if __name__ == '__main__':
    calc_acc(filename='weights_6040_True', isBi=True)
