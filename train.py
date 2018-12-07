# -*- coding: utf-8 -*-
# Kerasで自前のデータから学習と予測
#

import os
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.utils import np_utils
#from img_model import Img_Model
from img_loader import Img_Loader

#
#main
dir_path_train ="data/train"
dir_path_test  ="data/validation"
data = Img_Loader()
clsList = data.get_classes(dir_path_train )
(x_train, y_train) =data.get_data(dir_path_train)
(x_test, y_test)   =data.get_data(dir_path_test)

print(x_train.shape, y_train.shape )
print(x_test.shape   , y_test.shape )
print( clsList )
print( len(clsList ) )
num_classes =len(clsList )
#quit()

#画像を0-1の範囲で正規化
x_train=x_train.astype('float32')/255.0
x_test=x_test.astype('float32')/255.0

#正解ラベルをOne-Hot表現に変換
y_train=np_utils.to_categorical(y_train, num_classes)
y_test=np_utils.to_categorical(y_test, num_classes)

# model
#モデルを構築
model=Sequential()

model.add(Conv2D(32,(3,3),padding='same',input_shape=(128, 128,3)))
model.add(Activation('relu'))
model.add(Conv2D(32,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense( num_classes ,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

batch_size=32
epoch_num=20
#fit
history=model.fit(x_train,y_train
    ,batch_size=batch_size
    ,nb_epoch=epoch_num
    ,verbose=1,validation_split=0.1)

#モデルと重みを保存
json_string=model.to_json()
open('img_cnn.json',"w").write(json_string)
model.save_weights('img_cnn.h5')

#quit()
#print(y_train[:10])

