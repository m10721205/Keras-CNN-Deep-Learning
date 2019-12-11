import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPool2D
from keras.utils import np_utils

mnist = keras.datasets.mnist.load_data()
print(mnist)

# 輸入資料
(input_train, output_train), (input_test, output_test) = np.array(mnist)
print(input_train.shape)
print(output_train.shape)
print(input_test.shape)
print(output_test.shape)

# 整理資料型態
input_train = input_train.astype(dtype=np.float32)
input_test = input_test.astype(dtype=np.float32)

input_train = input_train.reshape(len(input_train), 28, 28, 1)
input_test = input_test.reshape(len(input_test), 28, 28, 1)

# 正規化
input_train /= 255  # 正規化0~1之間
input_test /= 255   # 正規化0~1之間

output_train = np_utils.to_categorical(output_train)  # 變成one-hot-encoding 的編碼方式
output_test = np_utils.to_categorical(output_test)    # 變成one-hot-encoding 的編號方式

# 以下是模型建立
model = Sequential()

model.add(Conv2D(filters=16,
                 kernel_size=(3,3),
                 activation='relu',
                 input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(x=input_train, y=output_train, epochs=10)