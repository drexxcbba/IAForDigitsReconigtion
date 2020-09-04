#In this project i'll use supervised learning.
#Importing what we need.
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.tree import DesicionTreeClasifier

#Reading the data.
data = pd.read_csv("train.csv").as_matrix()

#Creating the model to train.
treeModel = DesicionTreeClasifier() 

#Dividing our data in order to use it as a supervised learning.
x_train = data[0: 21000, 1:]
y_train = data[0: 21000, 0]

#Training
treeModel.fit(x_train, y_train)

#Predicting something.
x_test = data[21000:, 1:]
y_test = data[21000:, 0]
actual = x_test[8]
actual.shape(28, 28)
plt.imshow(255 - actual, cmap='gray')
print(treeModel.predict( [x_test[8]] ))
plt.show()

#Getting the accurancy.
res = treeModel.predict(x_test)
count = 0
for i in range(0, 21000):
    if res[i] == y_test[i]:
        count += 1
acc = (count / 21000) * 100
print("accurancy is :", acc)