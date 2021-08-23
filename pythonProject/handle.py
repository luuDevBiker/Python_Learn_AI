import cv2 as cv
import numpy as np
import matplotlib.pyplot as pl
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import os




# dir = r'~\PythonCode\Datasets' # path to folder datasets
# folder_image = ['zero','one','tow','three','four','five','six','seven','eight','nine']
# for folder in folder_image:
#     path = os.path.join(dir,folder)
# img = cv.imread(r'C:\Users\luubi\Documents\PythonCode\Datasets\one\0.png', cv.IMREAD_GRAYSCALE)
# cv.imshow('img',img)
# print(img[0][0])


mnist = tf.keras.datasets.mnist
#print(mnist)
(x_train , y_train) , (x_test , y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation =tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation =tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation =tf.nn.softmax))

model.compile(
                optimizer = 'adam',
                loss = 'sparse_categorical_crossentropy',
                metrics = ['accuracy']
                                )
model.fit(x_train ,y_train ,epochs = 2 )
model.save('train_letter.model')
new_model = tf.keras.models.load_model('train_letter.model')

val_loss , val_pass = model.evaluate(x_test,y_test)



img = cv.imread(r'~/PythonCode/0.png',0)
prediction = new_model.predict(x_test)
print(x_test[0])
print(np.argmax(prediction[0]))

print(val_pass , '\n' ,val_loss , '\n' , val_loss+val_pass)





