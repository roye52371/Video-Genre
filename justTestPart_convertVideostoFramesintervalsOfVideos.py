#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import math
import cv2
from pathlib import Path
import numpy as np
import shutil
from tqdm import tqdm as train_tqdm
from tqdm import tqdm as test_tqdm
from tqdm import trange

# if want to run different values, need to changed values in train part, and in test part afterwards!!!

# train part

#filename = 'Dataset70_30'  # OR "Dataset70_30"( to run second time with 'Dataset80_20')

# dataset_path = filename + "/train"  # dataset_path + train or test
# dataset_path_frames_dir = filename + "/train_frames_120perIntervalsOfVideo"
# # above is path will be created for new train/test dataset of frames intervals
# # below is creating the above folder
# os.makedirs(dataset_path_frames_dir, exist_ok=True)

classes = ["American Football", "Animation", "Baseball", "Basketball",
           "Cooking and food recipes", "Golf", "Graffiti Art", "Hair Style",
           "Ice Hockey", "Judo", "Soccer", "Speeches and Lectures-Talks",
           "Swimming(in Pool)", "Tennis", "Underwater-Ocean life",
           "Volleyball"]

#classes = ["Cooking and food recipes"]
# classes = ["American Football"]

seq_size = 2040

num_of_frames_per_intervals = 120

num_of_Intervals_per_Video = 17

# In[2]:


# # creat frames from train:
#
# # data_dir = dataset_path
# # Desktop/folder to split dataset videogenre try/Dataset70_30/train/Animation
# t_train_classes = train_tqdm(classes, desc='train dataset convert Progress Bar', unit="jenre", position=0, leave=True)
# for jenre in t_train_classes:
#     dataset_path_frames_jenre_dir = dataset_path_frames_dir + "/" + jenre
#
#     os.makedirs(dataset_path_frames_jenre_dir, exist_ok=True)
#
#     dataset_path_video_jenre = dataset_path + "/" + jenre
#     # print(dataset_path)
#
#     listing = os.listdir(dataset_path_video_jenre)
#     # print(listing)
#     # listing = [inst for inst in listing if not inst.startswith(".")]
#     # print(listing)
#     # seq_size = 120
#     for file in listing:
#         intervideoarray = []
#         # filenameeee= Path(file).stem
#         # print(filenameeee)
#         our_video = dataset_path_video_jenre + "/" + file
#         # print(our_video)
#         cap = cv2.VideoCapture(our_video)
#         # print(video.isOpened())
#         # framerate = video.get(5)
#         # os.makedirs(dataset_path_frames_jenre_dir + "/video_" + str(int(count)))
#         filenamenotype = Path(file).stem
#         # print(filenamenotype)
#         os.makedirs(dataset_path_frames_jenre_dir + "/video_" + filenamenotype, exist_ok=True)
#         # count = 0
#         if (cap.isOpened()):
#             # frameId = video.get(1)
#             # success,image = video.read()
#             numberofframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # frames size in video
#
#             # create all num of intervals videos to curr video
#             frame_width = int(cap.get(3))
#             frame_height = int(cap.get(4))
#             frampersec = 30  # or cap.get(cv2.CAP_PROP_FPS)# or 30
#             i_video_path = dataset_path_frames_jenre_dir + "/video_" + filenamenotype
#             for i in range(0, num_of_Intervals_per_Video):  # create intervals folder to video_i in it
#                 curr_interval_vid = i_video_path + "/" + str(i) + ".mp4"  # aka /video_i/intervalnumvid
#                 # out_curr_vid = cv2.VideoWriter(curr_interval_vid,cv2.VideoWriter_fourcc('F','M','P','4'), frampersec, (frame_width,frame_height))
#
#                 out_curr_vid = cv2.VideoWriter(curr_interval_vid, cv2.VideoWriter_fourcc(*'mp4v'), frampersec,
#                                                (frame_width, frame_height))
#                 intervideoarray.append(out_curr_vid)
#                 # out_curr_vid = cv2.VideoWriter(curr_interval_vid,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
#
#             # print(jenre)
#             # print(file)
#             # print(numberofframes)
#             jumping_frames = int(np.floor(numberofframes / seq_size))  # need to take frame after this number of times
#             # frame_index_array = []
#             for i in range(0,
#                            seq_size):  # data size is the video size, check if start from 0 or 1 and end with size or size+1
#                 # need to add index of frame  that devide with out reminder in self.seq_size from the specific video
#                 # cap.set(cv2.CAP_PROP_POS_FRAMES,i*jumping_frames)
#                 # changed to dumpy run over jumping frames, to prevent error accurs using cap.set in for loop
#                 for j in range(0, jumping_frames - 1):
#                     ret, frame = cap.read()
#                 ret, frame = cap.read()
#                 if ret == True:
#                     interval_to_move_frames_to = i % num_of_Intervals_per_Video  # modulo
#                     # i_video_path+"/interval_"+str(interval_to_move_frames_to)
#                     intervideoarray[interval_to_move_frames_to].write(frame)
#                     # resizearr = np.resize(frame, (3, 180, 220))
#                     # tensor[i] = torch.from_numpy(resizearr)
#                     # filename = dataset_path_frames_jenre_dir + "/video_" + filenamenotype + "/frame_" + str(int(count)) + ".jpg"
#                     # print(filename)
#                     # cv2.imwrite(filename,frame)
#                     # count = count+1
#                 else:
#                     print(
#                         "frame didnt extracted well or finished if shows uup try to delete this printing, in video:\n")
#                     print(our_video)
#                     break
#
#         else:
#             print("cap could not open - in video:\n")
#             print(our_video)
#             exit()
#         cap.release()
#
# #         i_video_path = dataset_path_frames_jenre_dir + "/video_" + filenamenotype
# #         for i in range(0, num_of_Intervals_per_Video): #create intervals folder to video_i in it
# #             curr_interval_folder= i_video_path+"/interval_"+str(i)
# #             os.makedirs(curr_interval_folder,exist_ok = True)
# # creating above all intervals folder in video_i
#
# #         for i in range(0, seq_size):
# #             interval_to_move_frames_to= i % num_of_Intervals_per_Video #modulo
#
# #             our_filename = dataset_path_frames_jenre_dir + "/video_" + filenamenotype + "/frame_" + str(int(i)) + ".jpg"
# #             curr_interval_folder= i_video_path+"/interval_"+str(interval_to_move_frames_to)
# #             check_existance= i_video_path+"/interval_"+str(interval_to_move_frames_to)+ "/frame_" + str(int(i)) + ".jpg"
# #             if os.path.exists(check_existance):
# #                os.remove(check_existance)
# #             shutil.move(our_filename,curr_interval_folder)


# In[3]:


# test part
filename = 'Dataset70_30'  # OR "Dataset70_30"( to run second time with 'Dataset80_20')
dataset_path = filename + "/test"  # os.path.join(filename, 'train')
dataset_path_frames_dir = filename + "/test_frames_120perIntervalsOfVideo"

os.makedirs(dataset_path_frames_dir, exist_ok=True)

# In[4]:


# creat frames from train:

# data_dir = dataset_path
# Desktop/folder to split dataset videogenre try/Dataset70_30/train/Animation
t_test_classes = test_tqdm(classes, desc='test dataset convert Progress Bar', unit="jenre", position=0, leave=True)
for jenre in t_test_classes:
    dataset_path_frames_jenre_dir = dataset_path_frames_dir + "/" + jenre

    os.makedirs(dataset_path_frames_jenre_dir, exist_ok=True)

    dataset_path_video_jenre = dataset_path + "/" + jenre
    # print(dataset_path)

    listing = os.listdir(dataset_path_video_jenre)
    # print(listing)
    # listing = [inst for inst in listing if not inst.startswith(".")]
    # print(listing)
    # seq_size = 120
    for file in listing:
        intervideoarray = []
        # filenameeee= Path(file).stem
        # print(filenameeee)
        our_video = dataset_path_video_jenre + "/" + file
        # print(our_video)
        cap = cv2.VideoCapture(our_video)
        # print(video.isOpened())
        # framerate = video.get(5)
        # os.makedirs(dataset_path_frames_jenre_dir + "/video_" + str(int(count)))
        filenamenotype = Path(file).stem
        # print(filenamenotype)
        os.makedirs(dataset_path_frames_jenre_dir + "/video_" + filenamenotype, exist_ok=True)
        # count = 0
        if (cap.isOpened()):
            # frameId = video.get(1)
            # success,image = video.read()
            numberofframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # frames size in video

            # create all num of intervals videos to curr video
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            frampersec = 30  # or cap.get(cv2.CAP_PROP_FPS)# or 30
            i_video_path = dataset_path_frames_jenre_dir + "/video_" + filenamenotype
            for i in range(0, num_of_Intervals_per_Video):  # create intervals folder to video_i in it
                curr_interval_vid = i_video_path + "/" + str(i) + ".mp4"  # aka /video_i/intervalnumvid
                # out_curr_vid = cv2.VideoWriter(curr_interval_vid,cv2.VideoWriter_fourcc('F','M','P','4'), frampersec, (frame_width,frame_height))

                out_curr_vid = cv2.VideoWriter(curr_interval_vid, cv2.VideoWriter_fourcc(*'mp4v'), frampersec,
                                               (frame_width, frame_height))
                intervideoarray.append(out_curr_vid)
                # out_curr_vid = cv2.VideoWriter(curr_interval_vid,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

            # print(jenre)
            # print(file)
            # print(numberofframes)
            jumping_frames = int(np.floor(numberofframes / seq_size))  # need to take frame after this number of times
            # frame_index_array = []
            for i in range(0,
                           seq_size):  # data size is the video size, check if start from 0 or 1 and end with size or size+1
                # need to add index of frame  that devide with out reminder in self.seq_size from the specific video
                # cap.set(cv2.CAP_PROP_POS_FRAMES,i*jumping_frames)
                # changed to dumpy run over jumping frames, to prevent error accurs using cap.set in for loop
                for j in range(0, jumping_frames - 1):
                    ret, frame = cap.read()
                ret, frame = cap.read()
                if ret == True:
                    interval_to_move_frames_to = i % num_of_Intervals_per_Video  # modulo
                    # i_video_path+"/interval_"+str(interval_to_move_frames_to)
                    intervideoarray[interval_to_move_frames_to].write(frame)
                    # resizearr = np.resize(frame, (3, 180, 220))
                    # tensor[i] = torch.from_numpy(resizearr)
                    # filename = dataset_path_frames_jenre_dir + "/video_" + filenamenotype + "/frame_" + str(int(count)) + ".jpg"
                    # print(filename)
                    # cv2.imwrite(filename,frame)
                    # count = count+1
                else:
                    print(
                        "frame didnt extracted well or finished if shows uup try to delete this printing, in video:\n")
                    print(our_video)
                    break

        else:
            print("cap could not open - in video:\n")
            print(our_video)
            exit()
        cap.release()

#         i_video_path = dataset_path_frames_jenre_dir + "/video_" + filenamenotype
#         for i in range(0, num_of_Intervals_per_Video): #create intervals folder to video_i in it
#             curr_interval_folder= i_video_path+"/interval_"+str(i)
#             os.makedirs(curr_interval_folder,exist_ok = True)
# creating above all intervals folder in video_i

#         for i in range(0, seq_size):
#             interval_to_move_frames_to= i % num_of_Intervals_per_Video #modulo

#             our_filename = dataset_path_frames_jenre_dir + "/video_" + filenamenotype + "/frame_" + str(int(i)) + ".jpg"
#             curr_interval_folder= i_video_path+"/interval_"+str(interval_to_move_frames_to)
#             check_existance= i_video_path+"/interval_"+str(interval_to_move_frames_to)+ "/frame_" + str(int(i)) + ".jpg"
#             if os.path.exists(check_existance):
#                os.remove(check_existance)
#             shutil.move(our_filename,curr_interval_folder)


# In[ ]:


# In[ ]:




