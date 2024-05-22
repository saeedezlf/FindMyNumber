# MNIST and Custom Digit Recognition

## Overview

This project focuses on training and evaluating different machine learning models for digit recognition using the MNIST dataset and a custom dataset created from a sample image. The models used include Perceptron, Logistic Regression, and Multi-layer Perceptron (MLPClassifier).

## Requirements

- Python 3.x
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - scikit-learn
  - cv2 (OpenCV)
  - mlxtend

## Dataset

### MNIST Dataset

- `mnist_train.csv`: Contains the training data for the MNIST dataset.
- `mnist_test.csv`: Contains the testing data for the MNIST dataset.

### Custom Dataset

A custom dataset is created by processing an image (`image0.jpeg`) containing handwritten digits from 0 to 9. The image is preprocessed, segmented into individual digits, and saved in `MyNumber.csv`.

## Steps

1. **Import Libraries and Load Data**

   Import necessary libraries and load the MNIST and custom datasets into pandas DataFrames.

2. **Image Preprocessing**

   - Load the image using OpenCV.
   - Convert the image to grayscale.
   - Invert the pixel values to make the digits white on a black background.
   - Crop and segment the image into individual digits.
   - Flatten each digit image into a 1D array and combine them into a dataset.
   - Save the custom dataset to `MyNumber.csv`.

3. **Load Custom Dataset**

   Load the custom dataset from `MyNumber.csv` and separate features and target values.

4. **Preprocess MNIST Dataset**

   - Extract features and target values from the MNIST training and testing datasets.
   - Standardize the features using `StandardScaler`.

5. **Train and Evaluate Models**

   Train and evaluate three different models on the standardized MNIST training data:
   
   - **Perceptron**
     - Train a Perceptron model.
     - Evaluate its accuracy on both MNIST test data and the custom dataset.
   
   - **Logistic Regression**
     - Train a Logistic Regression model.
     - Evaluate its accuracy on both MNIST test data and the custom dataset.
   
   - **MLPClassifier**
     - Train a Multi-layer Perceptron (MLP) model.
     - Evaluate its accuracy on both MNIST test data and the custom dataset.

## Results

The accuracy of each model on the MNIST test dataset and the custom dataset is printed to the console.

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import cv2

# Load MNIST datasets
df_train = pd.read_csv(r'C:\MachineLearning\5\archive\mnist_train.csv')
df_test = pd.read_csv(r'C:\MachineLearning\5\archive\mnist_test.csv')

# Image preprocessing
image = cv2.imread(r'C:\MachineLearning\6\image0.jpeg', cv2.IMREAD_GRAYSCALE)
image = np.where(image >= 100, 0, 256 - image)
image = image[70:420, 90:350]

# Segment the image into individual digits
a0 = image[10:70, 0:60]
a1 = image[110:170, 0:60]
a2 = image[210:270, 0:60]
a3 = image[290:350, 0:60]
a4 = image[10:70, 110:170]
a5 = image[100:160, 110:170]
a6 = image[190:250, 110:170]
a7 = image[290:350, 110:170]
a8 = image[10:70, 200:260]
a9 = image[100:160, 200:260]

# Flatten and stack the digit images
a0 = a0.flatten()
a1 = a1.flatten()
a2 = a2.flatten()
a3 = a3.flatten()
a4 = a4.flatten()
a5 = a5.flatten()
a6 = a6.flatten()
a7 = a7.flatten()
a8 = a8.flatten()
a9 = a9.flatten()

target = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape((-1, 1))
num = np.vstack((a0, a1, a2, a3, a4, a5, a6, a7, a8, a9))
num = np.hstack((num, target))
np.savetxt(r'C:\MachineLearning\6\MyNumber.csv', num, delimiter=',')

# Load custom dataset
df_mine = pd.read_csv(r'C:\MachineLearning\6\MyNumber.csv', header=None)
x_mine = df_mine.iloc[:, :784].values
y_mine = df_mine.iloc[:, -1].values

# Preprocess MNIST dataset
x_train = df_train.iloc[:, 1:].values.astype('float')
y_train = df_train.iloc[:, 0].values.astype('float')
x_test = df_test.iloc[:, 1:].values.astype('float')
y_test = df_test.iloc[:, 0].values.astype('float')

sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)
x_mine_std = sc.transform(x_mine)

# Train a Perceptron model on MNIST dataset
ppn = Perceptron(random_state=1)
ppn.fit(x_train_std, y_train)

# Predict on MNIST test dataset
y_pred_mnist = ppn.predict(x_test_std)
accuracy_mnist = accuracy_score(y_test, y_pred_mnist)
print("Accuracy on MNIST test dataset:", accuracy_mnist)

# Predict on custom number dataset
y_pred_mine = ppn.predict(x_mine_std)
accuracy_mine = accuracy_score(y_mine, y_pred_mine)
print("Accuracy on custom number:", accuracy_mine)

# Train a Logistic Regression model on MNIST dataset
lr = LogisticRegression(max_iter=100)
lr.fit(x_train_std, y_train)

# Predict on MNIST test dataset
y_pred_mnist_lr = lr.predict(x_test_std)
accuracy_mnist_lr = accuracy_score(y_test, y_pred_mnist_lr)
print("Accuracy on MNIST test dataset (Logistic Regression):", accuracy_mnist_lr)

# Predict on custom number dataset
y_pred_mine_lr = lr.predict(x_mine_std)
accuracy_mine_lr = accuracy_score(y_mine, y_pred_mine_lr)
print("Accuracy on custom number (Logistic Regression):", accuracy_mine_lr)

# Train a MLPClassifier model on MNIST dataset
mlp = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=100, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)
mlp.fit(x_train_std, y_train)

# Predict on MNIST test dataset
y_pred_mnist_mlp = mlp.predict(x_test_std)
accuracy_mnist_mlp = accuracy_score(y_test, y_pred_mnist_mlp)
print("Accuracy on MNIST test dataset (MLPClassifier):", accuracy_mnist_mlp)

# Predict on custom number dataset
y_pred_mine_mlp = mlp.predict(x_mine_std)
accuracy_mine_mlp = accuracy_score(y_mine, y_pred_mine_mlp)
print("Accuracy on custom number (MLPClassifier):", accuracy_mine_mlp)
```

## Conclusion

This project demonstrates the process of training and evaluating different machine learning models for digit recognition using both the MNIST dataset and a custom dataset. The results show the performance of each model on standard and custom data, highlighting the potential for custom digit recognition applications.
