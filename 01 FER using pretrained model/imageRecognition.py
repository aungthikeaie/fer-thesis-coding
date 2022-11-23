import tensorflow as tf
from tensorflow.keras.preprocessing import image # for preprocessing
from tensorflow.keras.applications import imagenet_utils # for prediction
import numpy as np
import matplotlib.pyplot as plt

filename = 'image_data//000032.jpg'

## first method to read an image (coudn't show result like jupyter notebook)
""" from IPython.display import Image
Image(filename = 'image_data//000031.jpg', width=224, height=224) """

## Second method to read an image 
""" from tensorflow.keras.preprocessing import image
img = image.load_img(filename, target_size=(224,224))
plt.imshow(img) """


## Third method to read an image
""" import cv2
img = cv2.imread(filename)
img = cv2.resize(img, (224,224))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) """

## Fourth method to read an image
""" from PIL import Image
img = Image.open(filename) ##PIL
img = img.resize((224,224))
plt.imshow(img) """

## Loading deep learning model
## creating a model
## training a model
## testing or validating - teach with labels
## prediction - let the deep learning guess

mobile = tf.keras.applications.mobilenet.MobileNet()
mobile2 = tf.keras.applications.mobilenet_v2.MobileNetV2()

# Preprocessing an image
img = image.load_img(filename, target_size=(224,224))
resized_image = image.img_to_array(img)
final_image = np.expand_dims(resized_image, axis=0)
final_image = tf.keras.applications.mobilenet.preprocess_input(final_image)

# final_image.shape (to view dimension)

# Using Mobile Net v1
predictions = mobile.predict(final_image)
results = imagenet_utils.decode_predictions(predictions)
print(results)

# Using Mobile Net v2
predictions2 = mobile2.predict(final_image)
results2 = imagenet_utils.decode_predictions(predictions2)
print(results2)

plt.imshow(img)




