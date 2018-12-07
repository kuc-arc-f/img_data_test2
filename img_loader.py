# -*- coding: utf-8 -*-
# モデルの定義

import os
import sys
import numpy as np

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
#
#dir_path_train ="data/train"
#dir_path_test  ="data/validation"
img_height, img_width = 128, 128
#
class Img_Loader:
        mClassList=[]
        def __init__(self ):
                param=0

        #
        def get_data(self , dir_base ):
                directory = os.listdir(dir_base )
                #print(directory)
                #mClassList=[]
                img_train =[]
                class_train=[]
                i=0
                for dname in directory:
#                        print('cat=' + dname)
#                        self.mClassList.append(i)
                        (img_train ,class_train )= self.get_images(dir_base + "/" +dname, i, img_train, class_train)
                        i+=1
                return np.array(img_train), np.array(class_train)
        #
        def get_images(self, dirname, idx_num ,img_dat, class_train ):
                #dirname= train_dir_path +"/" + base_dir
                for fnm in os.listdir(dirname):
                        #print(fnm)
                        img = image.load_img(dirname+ "/"+fnm, target_size=(img_height, img_width))
                        x = image.img_to_array(img)
                        #print(x.shape )
                        #x = np.expand_dims(x, axis=0)
                        #print(x.shape )
                        #quit()
                        #x = x / 255.0
                        img_dat.append(x)
                        class_train.append(idx_num)
                        #print(x)
                        #print(x.shape)
                        #quit()
                return img_dat, class_train
        #
        def get_classes(self, dir_base):
                self.mClassList= []
                directory = os.listdir(dir_base )
                #print(directory)
                i=0
                for dname in directory:
                        #print('cat=' + dname)
                        self.mClassList.append(dname )
                        i+=1
                return self.mClassList

