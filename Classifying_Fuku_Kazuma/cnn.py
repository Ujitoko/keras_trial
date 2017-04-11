import numpy as np
import os
from keras.preprocessing import image

# loadImages
def loadImages(dir:str):
    images = []
    files = []
    for f in os.listdir(dir):
        file = os.path.join(dir, f)
        img = image.load_img(file, target_size=(128,128))
        img_array = image.img_to_array(img)
        images.append(img_array)
        files.append(file)
    return (images, files)

#from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 40
num_classes = 2
epochs = 100
data_augmentation = True

# input image dimensions
img_rows, img_cols = 128, 128
# input image channels
img_channels = 3

#
script_path = os.path.dirname(__file__) # スクリプトのあるディレクトリへの相対パス
images_Fuku, files_Fuku = loadImages(os.path.join(script_path, 'Fuku_merge'))
images_Kazuma, files_Kazuma = loadImages(os.path.join(script_path, 'Kazuma_merge'))

images_Fuku = np.array(images_Fuku)
images_Kazuma = np.array(images_Kazuma)

print(type(images_Fuku))
print(type(images_Kazuma))

print(images_Fuku.shape)
print(images_Kazuma.shape)

print(len(images_Fuku))

# データサイズを合わせる
while(len(images_Fuku) > len(images_Kazuma)):
    i = np.random.randint(len(images_Fuku))
    images_Fuku = np.delete(images_Fuku, i, 0)
    files_Fuku = np.delete(files_Fuku, i, 0)
# print(images_Fuku.shape)
# print(images_Kazuma.shape)

# データを結合
x = np.vstack((images_Fuku, images_Kazuma))
print(x.shape)

# データを標準化
# sklearnのAPIには多次元配列は入力できないので、
# 一旦ベクトル化して処理した後、reshapeして多次元化する
from sklearn import preprocessing
x_tmp = x.reshape(x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3])
x_scaled = preprocessing.scale(x_tmp)
x_scaled = x_scaled.reshape(x.shape)
print(x_scaled.shape)

# ラベルデータを作成
num_data = len(images_Fuku)
y = np.hstack((np.zeros(num_data),np.ones(num_data)))

# ラベルデータをカテゴリ表現へ変換
y = keras.utils.to_categorical(y, num_classes)
# print(y)

# ホールドアウト検証のため、訓練データとテストデータに分割
from sklearn.cross_validation import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x_scaled, y, test_size = 0.2, random_state=11)
# print(x_train.shape)
# print(x_test.shape)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                padding='same',
                activation='relu',
                input_shape=x_train.shape[1:]))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, kernel_size=(3, 3),
                padding='same',
                activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.summary()

# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
