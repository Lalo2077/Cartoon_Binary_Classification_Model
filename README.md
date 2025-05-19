# Saiyan_Classification_Model
Supervised learning model for the classification of images of Saiyan transformations from Dragon Ball using Convolutional Neural Networks (CNNs).

# Dataset
The dataset consists of six classes:
* ssj - Super Saiyan
* ssj2 - Super Saiyan 2
* ssj3 - Super Saiyan 3
* ssj4 - Super Saiyan 4
* ssg - Super Saiyan God
* ssgss - Super Saiyan God Super Saiyan

The original dataset was sourced from kaggle, but it was too small. The first step was adding more images into it, I manually collected 80 new images and added them to each class, resulting in a balanced 100 images per class. 
The selection process was defined by two criterias:  
1. ¿Is it correct?
2. ¿Is it representative?
   
in the case for saiyan transformations, I prioritized images containing a single character to mantain visual clarity. If multiple characters were present, they should atleast have a higher percentage of saiyan characters in them. 

# Dataset Distribution 
The dataset is organized into two directories: *train* and *test*, each with six subdirectories.
| Class  | Train images | Test images |
| ------------- | ------------- | ------------- | 
| ssj  | 80 | 20
| ssj2  | 80 | 20
| ssj3  | 80 | 20
| ssj4  | 80 | 20
| ssg  | 80 | 20
| ssgss  | 80 | 20

Givin us a total of 480 train images and 120 test images

# Data preprocessing
For model robustness, the training images are going to be preprocessed using TensorFlow´s *ImageDataGenerator*. The steps are the following:

* Pixel normalization: rescale = 1./255 to normalize pixel values between 0 and 1.
* Rotation augmentation: rotation_range = 240° to create visual variability without losing key features.

Although other augmentation techniques like zoom and width range were tested, they were excluded due to many resulting augmented images cropped out essential visual features.

For example, some images ended up displaying only the body without the character´s distinct hair, which is a critical identifier in Saiyan transformations.

![image](https://github.com/user-attachments/assets/bd049b50-83ed-47ca-a7dc-3ebf5947b47a)

As a result, only rotation augmentation was retained to add variety to the training data. All images were resized to (150, 150) before being passed to the model.

# Examples of data augmentation 
![image](https://github.com/user-attachments/assets/f56642d7-deaf-44b8-867e-466f4f5ae6d1)

