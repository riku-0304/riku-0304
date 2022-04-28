from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from msilib.schema import Class
from pyexpat import model
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from tensorflow.keras.applications import EfficientNetV2B1
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import os
import glob
import shutil

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


label_dict = {
            0: 'India_10',
            1: 'India_20',
            2: 'India_50',
            3: 'India_100',
            4: 'India_200',
            5: 'India_500',
            6: 'India_2000',
            7: 'HongKong_10',
            8: 'HongKong_20',
            9: 'HongKong_50',
            10: 'HongKong_100',
            11: 'HongKong_500',
            12: 'HongKong_1000',
            13: 'Iraqi_200',
            14: 'Iraqi_500',
            15: 'Iraqi_1000',
            16: 'Iraqi_5000',
            17: 'Iraqi_1000',
            18: 'Nepal_500',
            19: 'Nepal_1000',
            20: 'Taiwan_100',
            21: 'Taiwan_500',
            22: 'Taiwan_1000',
            }

dir = 'val/20'
base_dir = 'currency_data'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
image_path = ''

batch_size = 16
image_shape = 224
num_classes = 23


def Split_Data():
    for num in range(num_classes):
        cl = str(num)
        img_path = os.path.join(base_dir, cl)                          # 取得單一類別資料夾路徑
        images = glob.glob(img_path + '/*.jpg')                        # 載入所有 jpg 檔成為一個 list
        print("{}: {} Images".format(cl, len(images)))                 # 印出單一類別有幾張圖片
        num_train = int(round(len(images) * 0.8))                        # 切割 80% 資料作為訓練集
        train, val = images[:num_train], images[num_train:]            # 訓練 > 0~80%，驗證 > 80%~100%

        for t in train:
            if not os.path.exists(os.path.join(base_dir, 'train', cl)):  # 如果資料夾不存在
                os.makedirs(os.path.join(base_dir, 'train', cl))           # 建立新資料夾
            shutil.move(t, os.path.join(base_dir, 'train', cl))          # 搬運圖片資料到新的資料夾

        for v in val:
            if not os.path.exists(os.path.join(base_dir, 'val', cl)):    # 如果資料夾不存在
                os.makedirs(os.path.join(base_dir, 'val', cl))             # 建立新資料夾
            shutil.move(v, os.path.join(base_dir, 'val', cl))            # 搬運圖片資料到新的資料夾


def GetData():
    image_gen_train = ImageDataGenerator(
        rotation_range=45,            # 隨機旋轉
        width_shift_range=0.2,        # 隨機水平移動
        height_shift_range=0.2,       # 隨機垂直移動
        zoom_range=0.3                # 隨機縮放
    )

    train_data_gen = image_gen_train.flow_from_directory(
        batch_size=batch_size,
        directory=train_dir,
        shuffle=True,
        target_size=(image_shape, image_shape),
        classes=[str(i) for i in range(num_classes)],
        class_mode='sparse'           # 分類標籤定義為 0, 1, 2, 3, 4
    )

    image_gen_val = ImageDataGenerator()

    val_data_gen = image_gen_val.flow_from_directory(
        batch_size=batch_size,
        directory=val_dir,
        target_size=(image_shape, image_shape),
        classes=[str(i) for i in range(num_classes)],
        class_mode='sparse'
    )
    return train_data_gen, val_data_gen


def Create_Model():
    dense_model = EfficientNetV2B1(include_top=False, input_shape=(224, 224, 3))
    t = dense_model.output
    t = GlobalAveragePooling2D()(t)
    t = Dense(1024, activation='relu')(t)
    predictions = Dense(num_classes, activation='softmax')(t)
    model = Model(inputs=dense_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    # print(model.summary())
    return model


def Training(model, train_data_gen, val_data_gen):
    history = model.fit(
        train_data_gen,
        epochs=10,
        validation_data=val_data_gen,
        verbose=1,
    )
    score = model.evaluate(val_data_gen, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return model, history


def Predict_single(model, image):
    img = load_img(image)
    img_array = img_to_array(img)
    img_array = tf.image.resize(img_array, [224, 224])
    test_x = np.array(img_array)
    test_x = np.expand_dims(test_x, 0)
    predict = model.predict(test_x)
    pred_y = predict.ravel()
    classes = np.argmax(predict, axis=1)
    t = []
    for i in classes:
        t.append(label_dict[i])
        country, value = t[0].split('_')
    print('Country:', country)
    print('Value:', value)
    plt.figure(figsize=(20, 5))
    plt.bar([i for i in range(len(pred_y))], pred_y, alpha=0.5, color='g')
    plt.xticks([i for i in range(len(pred_y))])
    plt.legend(['probability'], loc='upper left')
    plt.show()


def Predict_batch(model, dir):
    test_x = []
    for img_file in os.listdir(base_dir + '/' + dir):
        img = load_img(base_dir + '/' + dir + '/' + img_file)
        img_array = img_to_array(img)
        img_array = tf.image.resize(img_array, [224, 224])
        test_x.append(img_array)

    test_x = np.array(test_x)
    predict = model.predict(test_x)
    classes = np.argmax(predict, axis=1)
    print(classes)


def Save_Weights(model):
    model.save_weights('currency_weights.h5')


def Load_Weights(model):
    model.load_weights('currency_weights.h5')
    return model


def show_train_history(train_history):
    plt.plot(train_history.history['acc'])
    plt.plot(train_history.history['val_acc'])
    plt.xticks([i for i in range(0, len(train_history.history['acc']))])
    plt.title('Train History')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


train_data, val_data = GetData()


# model = Create_Model()
# model, history = Training(model, train_data, val_data)
# Save_Weights(model)
# show_train_history(history)


model = Create_Model()
model = Load_Weights(model)
Predict_batch(model, dir)
Predict_single(model, image_path)
