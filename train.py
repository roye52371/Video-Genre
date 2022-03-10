


import os
import glob
import  json
from HP_dataset import HP_Vit_Dataset
from torch.utils.data import Dataset,DataLoader
from vit_pytorch.vitHP import ViT_HP
import torch
import torch.optim as optim
import torch.nn as nn
if __name__ == '__main__':

    #### THIS APPLIED ONCE !!!!!
    # prepor_json()
    #################################
    # print("Remove Exit ..... You should not run from scratch !!")
    # exit(0)
    print("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    isXyOnly = False
    if isXyOnly == True:
        hand_points = 84
    else:
        hand_points = 126

    train_path = os.path.join('../Dataset_New/train_gaussian_gaussian')
    path_s = glob.glob(os.path.join(train_path,'*','*.txt'))
    thresh_samples = 30
    cnt=0
    th_paths=[]
    for pth in path_s:

        with open(pth, "r") as read_file:
            data = json.load(read_file)
            if thresh_samples <= len(data):
                cnt= cnt+1
                th_paths.append(pth)
                # min_frames = len(data)
            # print(len(data))


    batch_size=8
    train_dataset = HP_Vit_Dataset(th_paths,'../Dataset_New/classes.txt',thresh_samples=thresh_samples,hand_points=hand_points)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=0)

    for hp_data, label in train_loader:
        hp_data = hp_data.to(device)
        print(hp_data.shape)
        label = label.to(device)
        print(label.shape)
        exit(0)
    exit(0)
    test_path = os.path.join('../Dataset_New/test_gaussian_gaussian')
    path_s = glob.glob(os.path.join(test_path, '*', '*.txt'))
    th_paths = []

    for pth in path_s:

        with open(pth, "r") as read_file:
            data = json.load(read_file)
            if thresh_samples <= len(data):
                cnt = cnt + 1
                th_paths.append(pth)
                # min_frames = len(data)
            # print(len(data))

    test_dataset = HP_Vit_Dataset(th_paths, '../Dataset_New/classes.txt', thresh_samples=thresh_samples,
                                   hand_points=hand_points)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    num_frames = thresh_samples
    dim = hand_points

    for heads in [8,16,24]:
        for depth in [1,2,3]:
            for mlpdim in [1024,2048]:
                textTemplate = 'heads='+str(heads)+'_'+'depth='+str(depth)+'_'+'mlpdim='+str(mlpdim)+'_'+'isXyOnly='+str(isXyOnly)
                textTemplate_SAVE = 'vitHP_weights_' + textTemplate + '_accTest=' + str(0) + '.pth'
                # Plz need to changed


                net = ViT_HP(
                    num_patches = num_frames,
                    num_classes = 15,
                    dim = dim,
                    depth = depth,
                    heads = heads,
                    mlp_dim = mlpdim,
                    dropout = 0.1,
                    emb_dropout = 0.1
                )


                net = net.to(device)


                # input: the parameters to be optimized
                optimizer = optim.Adam(net.parameters(), lr=0.005)
                loss = nn.NLLLoss()

                # epochs and training
                epochs = 150
                trainLoss=0
                samples=0
                model_accuracy  =0
                for ep in range(epochs):
                    index = 1
                    total_loss = 0
                    data_cnt = 0
                    loss_out = 0
                    trainLoss = 0
                    samples = 0
                    for hp_data, label in train_loader:

                        hp_data = hp_data.to(device)
                        label = label.to(device)
                        output = net(hp_data)

                        loss_out = loss(output, label)

                        # if data_cnt % batch_size == 0 and data_cnt != 0:
                            # print('Progress =',data_cnt//batch_size,'/',len(train_loader)//batch_size)
                            # loss_out.backward: calculates the back-propagation algorithm
                        optimizer.zero_grad()
                        loss_out.backward()
                        optimizer.step()
                        # optimizes the model params

                        # update training loss, accuracy, and the number of samples
                        # visited
                        trainLoss += loss_out.item() * batch_size
                        samples += batch_size

                        # data_cnt = data_cnt + 1
                    trainTemplate = "epoch: {} train loss: {:.3f}"
                    print(trainTemplate.format(ep + 1, (trainLoss / samples)))
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
                    # data_cnt = data_cnt + 1
                    trainTemplate = "epoch: {} acc test: {:.3f}"
                    print(trainTemplate.format(ep + 1, accurcy / len(test_loader)))
                    acc_final = accurcy / len(test_loader)
                    if model_accuracy < acc_final:
                        model_accuracy = acc_final
                        print("We have new accuracy : ",model_accuracy)
                        if os.path.exists(textTemplate_SAVE)==True:
                            os.remove(textTemplate_SAVE)
                        textTemplate_SAVE = 'vitHP_weights_' + textTemplate + '_accTest=' + str(model_accuracy) + '.pth'
                        torch.save(net.state_dict(), textTemplate_SAVE)
                    net.train()
            #            print(textTemplate)
                #                 continue



                # simulated batch of images

            #