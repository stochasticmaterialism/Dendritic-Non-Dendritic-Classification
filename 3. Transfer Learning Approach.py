# The following code has been written in Google Colab

# Upload the dataset to a Google Drive and link Google Colab to that Google Drive account
from google.colab import drive  
drive.mount('/content/drive/')

# Insert name of your Google account
!mkdir "YYYYYYYYYYYYY" 

# XXXXXX implies name of the folder inside the Google drive where the dataset has been saved
!unzip "/content/drive/My Drive/XXXXXXXXXXX" -d YYYYYYYYYYYYY  

# Import necessary libraries
import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np 
import keras as keras
from keras.layers import GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import merge,Input
from keras.models import Model
from keras.utils import np_utils
from keras.models import load_model
from sklearn.metrics import accuracy_score
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as pi_incep
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input as pi_xcep 
import tensorflow as tf

# Each image will be reshaped to a dimension of 300*300
img_height=300
img_width=300

# The batch size is declared equal to the number of images used for training the pre trained networks, 70% of the total number of images
batch_size=2827

# Import the images with a 7:3 ratio between the training and validation sets
train_datagen=ImageDataGenerator(validation_split=0.3,preprocessing_function=preprocess_input) 
train_generator_vgg16=train_datagen.flow_from_directory("XXXXXX",target_size=(img_height,img_width),batch_size=batch_size,class_mode='categorical',subset='training') 
test_generator_vgg16=train_datagen.flow_from_directory("XXXXXX",target_size=(img_height,img_width),batch_size=batch_size,class_mode='categorical',subset='validation')

# For all the networks, max pooling, ADAM optimizer and cross entropy loss have been considered 

# Implementation of VGG16 network
new_input=Input(shape=(300,300,3))
model=VGG16(include_top=False,input_tensor=new_input,pooling='max',classes=2,input_shape=(300,300,3),weights='imagenet') # First we implement the netwrok with the ImageNet weights
#model.summary()
# We add fully connected and softmax layers at the end of the network
flat1=Flatten()(model.layers[-1].output)
class1=Dense(1024,activation='relu')(flat1)
output=Dense(2,activation='softmax')(class1)
model=Model(inputs=model.inputs,outputs=output)
# The netwrok architecture is checked
model.summary()
opt=Adam(lr=0.027) # Specify the learning the rate for the ADAM optimizer
model.compile(optimizer=opt,loss=keras.losses.categorical_crossentropy,metrics=['accuracy'])
model.summary()
# Try out training different combination of layers by freezing and unfreezing them 
for layer in model.layers[:0]:
	layer.trainable=False
for i,layer in enumerate(model.layers):
    print(i,layer.name,layer.trainable)
# After every iteration the weights will be saved only if the performace improves
checkpoint=ModelCheckpoint('model',monitor='val_accuracy',verbose=1,save_best_only=True,save_weights_only=False,mode='auto',period=1)
# Early stopping has been implemented to avoid unnecessary iterations
early=EarlyStopping(monitor='val_accuracy',min_delta=0,patience=20,verbose=1,mode='auto')
hist=model.fit_generator(steps_per_epoch=60,generator=train_generator_vgg16,validation_data=test_generator_vgg16,validation_steps=20,epochs=100,callbacks=[checkpoint,early])
# The model is saved and the performance is plotted
!mkdir -p saved_model
model.save('saved_model/Model_VGG16')
plt.plot(hist.history["accuracy"])
plt.plot(hist.history['val_accuracy'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Training Accuracy","Validation Accuracy","Training Loss","Validation Loss"])
plt.show()

# The implementation of InceptioV3 and Xception follow the same procedure as VGG16

# Implementation of Inception3
model1=InceptionV3(include_top=False,input_tensor=new_input,pooling='max',classes=2,input_shape=(300,300,3),weights='imagenet')
#model1.summary()
flat1=Flatten()(model1.layers[-1].output)
class1=Dense(1024,activation='relu')(flat1)
output=Dense(2,activation='softmax')(class1)
model1=Model(inputs=model1.inputs,outputs=output)
model1.summary()
opt=Adam(lr=0.339)
model1.compile(optimizer=opt,loss=keras.losses.categorical_crossentropy,metrics=['accuracy'])
model1.summary()
for layer in model1.layers[:203]:
	layer.trainable=False
for i,layer in enumerate(model1.layers):
    print(i,layer.name,layer.trainable)
checkpoint=ModelCheckpoint('model1',monitor='val_accuracy',verbose=1,save_best_only=True,save_weights_only=False,mode='auto',period=1)
early=EarlyStopping(monitor='val_accuracy',min_delta=0,patience=20,verbose=1,mode='auto')
hist1=model1.fit_generator(steps_per_epoch=60,generator=train_generator_vgg16,validation_data=test_generator_vgg16,validation_steps=20,epochs=100,callbacks=[checkpoint,early])
!mkdir -p saved_model
model.save('saved_model/Model_InceptionV3(203)')
# my_model directory
plt.plot(hist1.history["accuracy"])
plt.plot(hist1.history['val_accuracy'])
plt.plot(hist1.history['loss'])
plt.plot(hist1.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Training Accuracy","Validation Accuracy","Training Loss","Validation Loss"])
plt.show()

# Implementation of Xception
new_input=Input(shape=(300,300,3))
model2=Xception(include_top=False,input_tensor=new_input,pooling='max',classes=2,input_shape=(300,300,3),weights='imagenet')
#model2.summary()
flat1=Flatten()(model2.layers[-1].output)
class1=Dense(1024,activation='relu')(flat1)
output=Dense(2,activation='softmax')(class1)
model2=Model(inputs=model2.inputs,outputs=output)
model2.summary()
opt=Adam(lr=0.007)
model2.compile(optimizer=opt,loss=keras.losses.categorical_crossentropy,metrics=['accuracy'])
model2.summary()
for layer in model2.layers[:133]:
	layer.trainable=False
for i,layer in enumerate(model2.layers):
    print(i,layer.name,layer.trainable)
checkpoint=ModelCheckpoint('model2',monitor='val_accuracy',verbose=1,save_best_only=True,save_weights_only=False,mode='auto',period=1)
early=EarlyStopping(monitor='val_accuracy',min_delta=0,patience=20,verbose=1,mode='auto')
hist1=model2.fit_generator(steps_per_epoch=60,generator=train_generator_vgg16,validation_data=test_generator_vgg16,validation_steps=20,epochs=100,callbacks=[checkpoint,early])
!mkdir -p saved_model
model.save('saved_model/Model_Xception')
# my_model directory
!ls saved_model
plt.plot(hist1.history["accuracy"])
plt.plot(hist1.history['val_accuracy'])
plt.plot(hist1.history['loss'])
plt.plot(hist1.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Training Accuracy","Validation Accuracy","Training Loss","Validation Loss"])
plt.show()
