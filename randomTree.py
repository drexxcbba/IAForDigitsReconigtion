import matplotlib.pyplot as plt 
from sklearn.datasets import load_digits
import pylab as pl

digits = load_digits()

#Analyzing one image.
pl.gray()
pl.matshow(digits.images[0])
pl.show()

#Visualizing first 15 images with their labels.
data = list(zip(digits.images, digits.target))
plt.figure(figsize=(5, 5))
for item, (img, label) in enumerate(data[:15]):
    plt.subplot(3, 5, item + 1)
    plt.axis('off')
    plt.imgshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.tittle('%i' % label)

import random
from sklean import ensemble

#Dividing our data in order to use it as a supervised learning.
n = len(digits.images)
x = digits.images.reshape((n, -1))
y = digits.target

#Random indices.
sample_index = random.sample(range(len(x)), len(x) / 5)
valid_index = [i for i in range(len(x)) if i not in sample_index]

#Images and targets to work.
sample_images = [x[i] for i in sample_index] 
valid_images = [x[i] for i in valid_index]
sample_target = [y[i] for i in sample_index] 
valid_target = [y[i] for i in valid_index] 

#Creating the model to train.
tree_model = ensemble.RandomForestClassifier()

#Training.
tree_model.fit(sample_images, sample_target)

#Predict something and the accurancy.
res = tree_model.score(valid_images, valid_target)
print("accurancy is :", res)

