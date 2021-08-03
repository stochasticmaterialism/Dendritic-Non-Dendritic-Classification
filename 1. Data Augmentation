# The following code has been prepared using Jupyter Notebook

# Import necessary libraries
import cv2 
import matplotlib.pyplot as plt
from matplotlib import pyplot
import os
import glob
import numpy as np 
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array

#Load images
#Dendrites labelled as 1, non-dendrites as 0

img_dir=r"C:\Users\DEBANSHU BANERJEE\Desktop\dataset\dendritic"
data_path=os.path.join(img_dir,'*.jpg')
files=glob.iglob(data_path)
data,y,name=[],[],[]
# The name of the image files is imported along with the image
for f in files:
    s=" "
    if f[55]=='.':
        s=f[53:55]
    else:
        s=f[53:54]
    img=cv2.imread(f)
    #print(s)
    #print(f)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img=img[0:300,0:300]
    #plt.imshow(img,cmap='gray')
    #plt.show()
    data.append(img)
    y.append(1)
    name.append(s)
#print(len(data))
#print(len(y))

img_dir=r"C:\Users\DEBANSHU BANERJEE\Desktop\dataset\nondendritic"
data_path=os.path.join(img_dir,'*.jpg')
files=glob.iglob(data_path)
for f in files:
    img=cv2.imread(f)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img=img[0:300,0:300]
    #plt.imshow(img,cmap='gray')
    #plt.show()
    data.append(img)
    y.append(0)
#print(len(data))
#print(len(y))

'''for i in range(len(data)):
    plt.imshow(data[i])
    plt.show()'''
    
'''Keras ImageDataGenerator class provides a quick and easy way to augment your images. It provides a host of different augmentation techniques like standardization, rotation, shifts, flips, brightness change, and many more.
ImageDataGenerator class allows you to randomly rotate images through any degree between 0 and 360 by providing an integer value in the rotation_range argument.'''    
datagen=ImageDataGenerator(rotation_range=40,shear_range=.2,zoom_range=0.2,horizontal_flip=True,brightness_range=(0.5,1.5))
    
for i in range(len(data)): 
    #Converting the input sample image to an array
    x=img_to_array(data[i])
    x=x.reshape((1,)+x.shape)
    #Generating and saving 5 augmented samples using the above defined parameters. 
    if (y[i]==1):
        j=0
        for batch in datagen.flow(x,batch_size=1,save_to_dir=r'C:\Users\DEBANSHU BANERJEE\Desktop\Dnd_Gray\dendritic',save_prefix=name[i]+'g'+str(j),save_format='jpg'):
            j+=1
            if j>5:
                break
    else:
        j=0
        for batch in datagen.flow(x,batch_size=1,save_to_dir=r'C:\Users\DEBANSHU BANERJEE\Desktop\DnD_Gray\nondendritic',save_prefix='image',save_format='jpg'):
            j+=1
            if j>5:
                break
