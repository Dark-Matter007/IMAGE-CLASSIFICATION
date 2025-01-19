import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets,layers,models

(training_images,training_labels),(testing_images,testing_labels) = datasets.cifar10.load_data()
training_images , testing_images = training_images / 255, testing_images / 255, 


class_names=['plane','car', 'bird', 'cat', 'deer' , 'dog', 'frog' , 'horse', 'ship', 'truck']


model = models.load_model('image_classifier.model')

img = cv.imread('images/frog.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)

prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction) #gives us maximum value or maximum neuron
print(f'prediction is {class_names[index]}')


plt.show()


