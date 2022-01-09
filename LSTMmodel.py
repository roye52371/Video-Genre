from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torchviz import make_dot
import torch.nn.functional as F

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)


class FeatureExtractors(nn.Module):
    """
    The Extract an feature vector
    """

    def __init__(self, model_name):
        """
        Initialize the model by setting up the layers.
        """
        super(FeatureExtractors, self).__init__()

        try:
            if model_name == 'vgg16':
                self.md_ftextractor = models.vgg16(pretrained=True)
            elif model_name == 'vgg19':
                self.md_ftextractor = models.vgg19(pretrained=True)
            elif model_name == 'vgg16_bn':
                self.md_ftextractor = models.vgg16_bn(pretrained=True)
            elif model_name == 'vgg19_bn':
                self.md_ftextractor = models.vgg19_bn(pretrained=True)

            elif model_name == 'resnet101':
                self.md_ftextractor = models.resnet101(pretrained=True)

            elif model_name == 'resnet152':
                self.md_ftextractor = models.resnet152(pretrained=True)

            elif model_name == 'densenet121':
                self.md_ftextractor = models.densenet121(pretrained=True)

            elif model_name == 'densenet169':
                self.md_ftextractor = models.densenet169(pretrained=True)

            # elif model_name == 'resnext50':
            #     self.md_ftextractor = models.resnext50(pretrained=True)
            # elif model_name == 'resnext101':
            #     self.md_ftextractor = models.resnext101(pretrained=True)

            self.model_name = model_name
            self.output_size = list(self.md_ftextractor.parameters())[-1].size()
        except:
            print("The model is not exist, plz add to init_dictonary_models")

    def forward(self, x):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        return self.md_ftextractor(x)


class MyBasicModel(nn.Module):
    expansion = 1

    def __init__(self, channel_in, latent_dim=256, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(MyBasicModel, self).__init__()
        self.channel_in = channel_in

        self.layer1 = nn.Sequential(
            nn.Conv2d(self.channel_in, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, latent_dim, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(256, output_size)
        self.output_size = latent_dim

    def forward(self, x):
        # ours x.size: [4, 3, 222, 180]
        # example x.size() [8, 3, 180, 220]
        try:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
        except Exception as e:
            print(np.array(x).shape)
            raise e
        # x = self.fc(x)
        return x


class LSTMmodel(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, output_size, latent_dim, hidden_dim, n_layers, drop_prob=0.5, model_name='Basic', isBi=False,
                 channel=3):
        """
        Initialize the model by setting up the layers.
        """
        super().__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.isBi = isBi
        self.channel = channel

        if model_name == 'Basic':
            self.featureExtractor = MyBasicModel(channel_in=self.channel, latent_dim=latent_dim)
        else:
            print("There is no name like this")
            exit(0)
        # # FEATURE EXTRACTOR + INPUT_SIZE
        # self.featureExtractor =  FeatureExtractors(model_name=model_name)
        self.input_size = self.featureExtractor.output_size
        self.latent_dim = latent_dim
        # print("The out size of model feature extractor", self.input_size )
        # self.fcDecod = nn.Linear( self.input_size[0],  self.latent_dim)
        self.model_name = model_name

        # output-size equal (1,2) laugh or not !
        # self.input_size  : vector extracted size from feature extractor
        # hidden_dim : the hidden output from each cell in LSTM

        #         #1000x1
        if isBi is False:
            self.lstm = nn.LSTM(self.latent_dim, hidden_dim, n_layers,
                                dropout=drop_prob, batch_first=False)
        else:
            self.lstm = nn.LSTM(self.latent_dim, hidden_dim, n_layers,
                                dropout=drop_prob, batch_first=False, bidirectional=True)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layers
        if isBi is False:
            self.fc1 = nn.Linear(self.hidden_dim, 128)
        else:  # BI
            self.fc1 = nn.Linear(2 * self.hidden_dim, 128)
        self.fc2 = nn.Linear(128, output_size)

        ## could manupliate the deccoders
        # self.sig = nn.Sigmoid()
        self.sig = nn.Softmax(dim=1)

    def config_(self):
        if self.isBi is False:
            return self.model_name + 'LSTM' + "_" + "L_" + str(self.latent_dim) + "_" + "H_" + str(
                self.hidden_dim) + "_" + "BiDir=False"
        else:
            return self.model_name + 'LSTM' + "_" + "L_" + str(self.latent_dim) + "_" + "H_" + str(
                self.hidden_dim) + "_" + "BiDir=True"

    def forward(self, x):
        """
        BxTxCxWxH(tuple): # x = torch.rand(1, 10, 3, 180, 220)
            B: batch size
            T: time series (no. of frames)
            C: channel
            W: width
            H: height
        """
        # ours x.size()= [4, 50, 3, 222, 180]
        # example x.size()= [8, 10, 3, 180, 220]

        hidden = None

        # iterates over all the frames(T) and in each frame calculates the hidden(characteristics of the frames) and in
        # the end of for loop, saves the lstm_out
        for t in range(x.size(1)):  # move over frames for each sample
            fetEx = self.featureExtractor(x[:, t, :, :, :])
            lstm_out, hidden = self.lstm(fetEx.unsqueeze(0), hidden)

        out = self.fc1(lstm_out[-1, :, :])
        out = F.relu(out)
        out = self.fc2(out)
        out = self.sig(out)

        return out  # Bx2


# import matplotlib.pyplot as plt
# from tqdm import tqdm
#
if __name__ == '__main__':
    # Instantiate the model w/ hyperparams
    output_size = 16  # classification output in our case 16
    hidden_dim = 256  # no of
    latent_dim = 512
    n_layers = 1
    batch_size = 1
    seq = 30

    net = LSTMmodel(output_size, latent_dim, hidden_dim, n_layers, model_name='Basic', isBi=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)
    x = torch.rand(batch_size, seq, 3, 180, 220)# 3 is rgb, (180,220) specific frame size
    for tensor, lbl in datasets:
        tensor = tensor.to(device)
        y = net(tensor)
        print(y)

    dot = make_dot(y)
    dot.format = 'pdf'
    dot.render()
