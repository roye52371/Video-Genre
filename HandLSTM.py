from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
#from torchviz import make_dot
import torch.nn.functional as F


class LSTMmodel(nn.Module):
    def __init__(self, output_size, latent_dim, hidden_dim, n_layers, channel=3, drop_prob=0.5, isBi=False,
                 model_name='Media Basic'):
        """
        # latent_dim -> input of lstm (input_dim)  input of dimension 5 will look like this [1, 3, 8, 2, 3]
        # hidden_dim -> the vector length received from the past lstm run
        """
        super(LSTMmodel, self).__init__()

        self.output_size = output_size

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.channel = channel
        self.isBi = isBi
        self.model_name = model_name

        self.lstm = nn.LSTM(latent_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True, bidirectional=isBi)
        # dropout layer
        self.dropout = nn.Dropout(0.3)

        if isBi is False:
            self.fc1 = nn.Linear(self.hidden_dim, 128)
        else:  # BI
            self.fc1 = nn.Linear(2 * self.hidden_dim, 128)
        self.fc2 = nn.Linear(128, output_size)

        ## could manupliate the deccoders
        # self.sig = nn.Sigmoid()
        self.sig = nn.LogSoftmax(dim=1)

    def config_(self):
        return self.model_name + 'LSTM' + "_" + "L_" + str(self.latent_dim) + "_" + "H_" + str(
            self.hidden_dim) + "_" + f"BiDir={self.isBi}"

    def forward(self, lm_seq):
        """
        the input of the model is the landmarks of the hand detected in the frames
        frame: hand: landmark: coordinate

        BxTX1X42: # x = torch.rand(1,30,1,42)
            B: batch size
            T: time series (no. of landmarks)
        """
        lstm_out = None
        hidden = None

        for t in range(lm_seq.size(1)):  # move over frames for each sample
            lstm_out, hidden = self.lstm(lm_seq[:, t, :, :], hidden)

        lstm_out = lstm_out[:, -1]
        out = self.fc1(lstm_out)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.sig(out)

        return out  # Bx2

# if  __name__ == '__main__':
#
#     output_size = 10
#     # the vector length received from the past lstm run
#     hidden_dim = 256
#     # input of lstm
#     # todo: change this value
#     latent_dim = 42
#     # how many frames to look back (check it)
#     n_layers = 1
#
#     lstm = LSTMmodel(output_size, latent_dim, hidden_dim, n_layers, model_name='Basic', isBi=False)
#     # use the available device in the model
#     x = torch.rand(5, 10, 1, 42)
#     out = lstm(x)
