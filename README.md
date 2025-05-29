# Cartoon_Binary_Classification_Model

*José Eduardo Díaz Maldonado - A01735676

Supervised learning model for the binary classification of images of the cartoon characters Tom and Jerry using Convolutional Neural Networks (CNNs). The original idea of multiclassification of saiyan characters from Dragon Ball was scrapped.

# Dataset
The dataset consists two classes:
* Tom 
* Jerry

The original dataset was sourced from kaggle, the dataset contained originally 5478 image files divided in four directories: jerry, tom, tom_jerry_0 and tom_jerry_1, I decided to reduce it to 2400, because some images didn´t contain neither of the characters 'tom' or 'jerry'.
The selection process was defined by two criterias:  
1. ¿Is it correct?
2. ¿Is it representative?
   
With this reasoning, most images with other characters different from the two classes were discarded.

# Dataset Distribution 
The dataset is organized into three directories: *train*, *validation* and *test*, each with two categories.

| Class  | Train images | Validation Images | Test images |
| ------------- | ------------- | ------------- | 
| tom | 800 | 180 | 180 
| jerry  | 800 | 180 | 180

Givin us a total of 1680 train images, 360 validation images and 360 test images.

# Data preprocessing
For model robustness, the training images are going to be preprocessed using TensorFlow´s *ImageDataGenerator*. The steps are the following:

* Pixel normalization: rescale = 1./255 to normalize pixel values between 0 and 1.
* Rotation augmentation: rotation_range = 240° to create visual variability by moving the image without losing key features.

All images were resized to (150, 150) before being passed to the model.

# Examples of data augmentation 






