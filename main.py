import cv2
import numpy as np
import os


def add(train_filepath, label_filepath):
    train = []
    label = []
    train_list = os.listdir(train_filepath)
    label_list = os.listdir(label_filepath)

    train_list.sort()
    label_list.sort()

    for i in range(len(train_list)):

        if (train_list[i] != label_list[i]):
            print(i)
        img_train = cv2.cvtColor(cv2.imread(train_filepath + '/' + train_list[i]), cv2.COLOR_BGR2GRAY)
        img_label = cv2.cvtColor(cv2.imread(label_filepath + '/' + label_list[i]), cv2.COLOR_BGR2GRAY);
        train.append(np.resize(img_train, (512, 512, 1)));
        label.append(np.resize(img_label, (512, 512, 1)));
    return train, label

train = [];
label = [];

train_input = [ '/home/hzk/Desktop/IM_finalproject/Data/data01/image',
               '/home/hzk/Desktop/IM_finalproject/Data/data02/image',
               '/home/hzk/Desktop/IM_finalproject/Data/data03/image',
               '/home/hzk/Desktop/IM_finalproject/Data/data04/image']
label_input = ['/home/hzk/Desktop/IM_finalproject/Data/data01/label',
              '/home/hzk/Desktop/IM_finalproject/Data/data02/label',
              '/home/hzk/Desktop/IM_finalproject/Data/data03/label',
              '/home/hzk/Desktop/IM_finalproject/Data/data04/label']

for i in range(4):
    t , l = add(train_input[i] , label_input[i])
    train.append(t)
    label.append(l)

from util import unet


from keras import Input


net = unet()
label = np.array(label[0])
train = np.array(train[0])
net.fit( train, label , batch_size = 10 , epochs = 5 )