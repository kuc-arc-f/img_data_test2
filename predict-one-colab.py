# -*- coding: utf-8 -*-
# 画像検証の処理
# 起動: python predit.py filename
# (ex: python predit.py cat.jpg )

import os
import sys
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.models import model_from_json
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing import image
import numpy as np
from keras.layers import Convolution2D, MaxPooling2D, Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
#from keras.applications.inception_v3 import preprocess_input, decode_predictions, InceptionV3
#from img_model import Img_Model
from img_loader import Img_Loader

#main
#arg =sys.argv[1]
#filename="data/validation/cat/neko_24.jpg"
#filename="data/validation/dog/dog_11.jpg"
filename="data/validation/bird/tori_11.jpg"
print(filename)
#quit()
#
#batch_size = 32

#img-load
img_height, img_width = 128, 128
# 画像を読み込んで4次元テンソルへ変換
img = image.load_img(filename, target_size=(img_height, img_width))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
# 学習時にImageDataGeneratorのrescaleで正規化したので同じ処理が必要
x = x / 255.0
#print(x)
#print(x.shape)

#
#data
#main
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

# クラスを予測, 入力は1枚の画像なので[0]のみ
#load
model_file = "img_cnn.json"
with open(model_file, 'r') as fp:
    model = model_from_json(fp.read())
model.load_weights("img_cnn.h5")
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

pred  = model.predict(x)
print(pred[0])
p_idx= np.argmax(pred[0] ) # 最も確率の高い要素のインデックスを取得
print(p_idx )
print(clsList[p_idx])
