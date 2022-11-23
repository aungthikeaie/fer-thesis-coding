#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf ##pip install tensorflow


# In[2]:


import numpy as np


# In[3]:


filename = 'image_data//000031.jpg'


# # First method to read an image (from storage to RAM)

# In[14]:


from IPython.display import Image
Image(filename = 'image_data//000031.jpg', width=224, height=224)


# # Second method to read an image

# In[5]:


from tensorflow.keras.preprocessing import image
img = image.load_img(filename, target_size = (224,224))


# In[6]:


import matplotlib.pyplot as plt ##conda install -c conda-forge matplotlib


# In[7]:


plt.imshow(img)


# # Third method to read an image

# In[10]:


import cv2 ## pip install opencv-python


# In[13]:


imgg = cv2.imread(filename) #read image as BGR not RGB
imgg = cv2.resize(imgg, (224,224))
plt.imshow(cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB)) #convert BGR to RGB


# # Fourth method to read an image

# In[15]:


from PIL import Image ## pip install Pillow


# In[18]:


im = Image.open(filename) ##PIL
im = im.resize((224,224))
plt.imshow(im)


# # Loading the deep learning model

# In[8]:


mobile = tf.keras.applications.mobilenet.MobileNet() ## deep learning model weights - pre-trained


# In[37]:


## creating a model
## training a model
## testing or validating - teach with labels
## prediction - let the deep learning guess


# In[38]:


mobile2 = tf.keras.applications.mobilenet_v2.MobileNetV2()


# # Preprocessing an image

# In[36]:


from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

filename = 'image_data//000031.jpg'
img = image.load_img(filename, target_size = (224, 224))
plt.imshow(img)


# In[27]:


resized_img = image.img_to_array(img)
final_image = np.expand_dims(resized_img, axis = 0) #adding fourth dimension
final_image = tf.keras.applications.mobilenet.preprocess_input(final_image)


# In[29]:


final_image.shape


# In[30]:


predictions = mobile.predict(final_image)


# In[32]:

from tensorflow.keras.applications import imagenet_utils


# In[33]:


results = imagenet_utils.decode_predictions(predictions) # decode predictions into readable format


# In[34]:


print(results)


# In[35]:


plt.imshow(img)


# # Using Mobile Net V2

# In[42]:


predictions2 = mobile2.predict(final_image)


# In[43]:


results2 = imagenet_utils.decode_predictions(predictions2)


# In[44]:


print(results2)


# In[ ]:




