#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import splitfolders
#path = 'C:\Users\roi52\Desktop\folder so split dataset videogenre try\partial_dataset'
#dirname= os.path.dirname(path)
#70_30
input_folder = 'Dataset/'
splitfolders.ratio(input_folder, output="Dataset70_30", seed=1337, ratio=(0.7, 0, 0.3), group_prefix=None)

#80_20
#path = 'C:\Users\roi52\Desktop\folder so split dataset videogenre try\partial_dataset'
#dirname= os.path.dirname(path)
input_folder = 'Dataset/'
splitfolders.ratio(input_folder, output="Dataset80_20", seed=1337, ratio=(0.8, 0, 0.2), group_prefix=None)


# In[ ]:




