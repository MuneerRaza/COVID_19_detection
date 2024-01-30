# COVID-19 Radiography Dataset Classification

## Overview
This project aims to classify X-ray images from the COVID-19 Radiography Dataset into two categories: Normal and COVID-19 positive. The classification is performed using a Convolutional Neural Network (CNN) implemented in TensorFlow and Keras. The dataset contains X-ray images from individuals with and without COVID-19.

## Required Libraries
- Pandas
- NumPy
- Seaborn
- Matplotlib
- OpenCV
- Scikit-learn
- TensorFlow

Install the required libraries using:
```bash
pip install pandas numpy seaborn matplotlib opencv-python scikit-learn tensorflow
```

## Data Extraction
The dataset is stored in the directory specified by the 'path' variable. Two subdirectories, 'COVID' and 'Normal', contain X-ray images of individuals with COVID-19 and healthy individuals, respectively.

## Imbalanced Dataset
The number of images in the 'COVID' and 'Normal' folders is checked, revealing an imbalanced dataset. Balancing the dataset is crucial for unbiased model training.

## Image Exploration
A sample image is loaded, and its dimensions, size, and RGB values are explored using OpenCV and Matplotlib. The image is not resized, as all images in the dataset are already 299x299 pixels.

## Augmentation Techniques
Various augmentation techniques, including Random Fog, Random Brightness, Random Crop, Rotate, RGB Shift, Vertical Flip, and Random Contrast, are applied to a chosen COVID-19 image to showcase different augmentation effects.

## Data Loading and Preprocessing
Images and labels are loaded for both COVID-19 and Normal classes. The data is normalized, and due to memory constraints, images are resized to 100x100 pixels. The dataset is split into training, validation, and test sets.

## Model Architecture
A simple CNN model is defined using TensorFlow and Keras, consisting of convolutional layers, max-pooling layers, and dense layers. The model is compiled with the Adam optimizer and binary crossentropy loss.

## Model Training
The CNN model is trained on the training set with explicit validation data. The training process is visualized using Matplotlib.

## Model Evaluation
The trained model is evaluated on the test set, and the test accuracy and loss are reported. Additionally, confusion matrices and classification reports are generated for the training, validation, and test sets.

## Results Visualization
Training and validation accuracy, loss, and confusion matrices are visualized using Matplotlib and Seaborn.

## Conclusion
The project provides a simple yet effective approach to classify COVID-19 and normal X-ray images. The implemented CNN model demonstrates good performance on the test set, achieving a high accuracy while avoiding overfitting. Further improvements can be explored, such as fine-tuning the model architecture or experimenting with additional data augmentation techniques.
