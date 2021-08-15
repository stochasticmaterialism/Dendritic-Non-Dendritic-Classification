# The following code has been prepared using Jupyter Notebook

# Import necessary libraries
import cv2 
import matplotlib.pyplot as plt
import os
import glob
import numpy as np 
import pandas as pd
import keras as keras
from keras.layers import GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as pi_incep
from keras.preprocessing import image
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import merge,Input
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.models import Sequential

#Load images
#Dendrites labelled as 1, non-dendrites as 0

img_dir=r"C:\Users\DEBANSHU BANERJEE\Desktop\dataset\dendritic"
data_path=os.path.join(img_dir,'*.jpg')
files=glob.iglob(data_path)
data,y=[],[]
for f in files:
    img=cv2.imread(f)
    #plt.imshow(img)
    #plt.show()
    #img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img=img[0:300,0:300]
    data.append(img)
    y.append(1)
print("Dendritic images = {}".format(len(data)))
#print(len(y))

img_dir=r"C:\Users\DEBANSHU BANERJEE\Desktop\dataset\nondendritic"
data_path=os.path.join(img_dir,'*.jpg')
files=glob.iglob(data_path)
for f in files:
    img=cv2.imread(f)
    #plt.imshow(img)
    #plt.show()
    #img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img=img[0:300,0:300]
    data.append(img)
    y.append(0)
print("Non dendritic images = {}".format(len(data)))
#print(len(y))

'''for i in range(len(data)):
    plt.imshow(data[i])
    plt.show()'''
    
# Implement Inception V3
model=InceptionV3(weights='imagenet',include_top=False)
model=Model(inputs=model.inputs,outputs=model.layers[-2].output)
model.summary()

# Values from the penultimate fully connected layer is considered as the features
f_incep=[]
for i in range(len(data)):
    f=[]
    img=data[i]
    img_data=image.img_to_array(img)
    img_data=np.expand_dims(img_data,axis=0)
    img_data=preprocess_input(img_data)
    incep_feature=model.predict(img_data)
    incep_feature=incep_feature.flatten()
    #print(incep_feature)
    print(incep_feature.shape)
    f_incep.append(incep_feature)
    

f_incep=np.array(f_incep)
print(f_incep.shape)

# The feature set is saved as CSV file
x=pd.DataFrame(f_incep)
x.to_csv('name.csv')
