# -*- coding: utf-8 -*-
# 評価。

import numpy as np
import os
import sys
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers.core import Dense,Activation,Dropout,Flatten
from keras.datasets import cifar10
from keras.utils import np_utils
from img_loader import Img_Loader

if __name__ == '__main__':
    dir_path_train ="data/train"
    dir_path_test  ="data/validation"
    data = Img_Loader()
    clsList = data.get_classes(dir_path_train )
    (x_train, y_train) =data.get_data(dir_path_train)
    (x_test, y_test)   =data.get_data(dir_path_test)

    print(x_train.shape, y_train.shape )
    print(x_test.shape   , y_test.shape )
    print( clsList[0] )
    print( len(clsList ) )
    num_classes =len(clsList )

    #画像を0-1の範囲で正規化
    x_train=x_train.astype('float32')/255.0
    x_test=x_test.astype('float32')/255.0

    #正解ラベルをOne-Hot表現に変換
    y_train=np_utils.to_categorical(y_train, num_classes)
    y_test=np_utils.to_categorical(y_test, num_classes)
#    quit()

    #load
    model_file = "img_cnn.json"
    with open(model_file, 'r') as fp:
        model = model_from_json(fp.read())
    model.load_weights("img_cnn.h5")
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#    model.summary()

    # モデルの評価
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', loss)
    print('Test acc:', acc)
    print("#end_acc")

    #画像を予想
    #img_pred=model.predict_classes(temp_img_array)
    #print('\npredict_classes=',img_pred)
