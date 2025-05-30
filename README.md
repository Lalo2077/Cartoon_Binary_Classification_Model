# Cartoon_Binary_Classification_Model

*José Eduardo Díaz Maldonado - A01735676*

Supervised learning model for the binary classification of images of the cartoon characters Tom and Jerry using Convolutional Neural Networks (CNNs). The original idea of multiclassification of saiyan characters from Dragon Ball was scrapped.

# Dataset
The dataset consists two classes:
* Tom 
* Jerry

The original dataset was sourced from kaggle, the dataset contained originally 5478 image files divided in four directories: jerry, tom, tom_jerry_0 and tom_jerry_1, It was decided to reduce it to 2400, because some images didn´t contain neither of the characters 'tom' or 'jerry'.
The selection process was defined by two criterias:  
1. ¿Is it correct?
2. ¿Is it representative?
   
With this reasoning, most images with other characters different from the two classes were discarded.

# Dataset Distribution 
The dataset is organized into three directories: *train*, *validation* and *test*, each with two categories.

| Class  | Train images | Validation Images | Test images |
| ------------- | ------------- | ------------- | ------------- | 
| tom | 800 | 180 | 180 
| jerry  | 800 | 180 | 180

Givin us a total of 1680 train images, 360 validation images and 360 test images.

# Data preprocessing
For model robustness, the training images are going to be preprocessed using TensorFlow´s *ImageDataGenerator*. The steps are the following:

* Pixel normalization: rescale = 1./255 to normalize pixel values between 0 and 1.
* Rotation augmentation: rotation_range = 240° to create visual variability by moving the image without losing key features.

All images were resized to (150, 150) before being passed to the model.

# Examples of data augmentation 
![image](https://github.com/user-attachments/assets/804ef9b7-d793-4876-bd93-f835aa26517e)

# Model 
The model uses the concept of transfer learning for the base, it uses the pre-trained model called VGG16, which contains 16 layers between convolutional and max-pooling. The final layers are discarted using "include_top= False" because the original model was designed to classify  1000 classes from the ImageNet dataset. Additionaly, the trainable qualities of VGG16 are stopped with "trainable = False". We don´t need it to learn from this project´s dataset, we only take advantage of the viusal features it already knows how to detect.

New layers were added to replace the original final layers and specialize the model for its specific classification task. A Flatten layer converts the feature maps into a one-dimension array, then it goes through a Dense layer with 256 nodes that uses ReLu activation. Finally, a single neuron wih sigmoid activation is used for the output probability between 0 and 1, representing one of the two possible classes.

The model is compiled with the RMSprop optimizer and binary_crossentropy as the loss function. It is then trained for 10 epochs.
![image](https://github.com/user-attachments/assets/396a6c9e-ce83-43f8-83ea-d9dc630f2b1a)

# Results
![image](https://github.com/user-attachments/assets/0bf6a70d-8869-4ab2-84b9-662ae829c982)

The training accuracy starts low with 57.8% but grows steadily to 87.3% by epoch 10, meaning that the model is learning, but when it comes to validation it starts higher than training with 62.5% but as the epochs go through, it stays and doesn´t increase as much as training. The interpretation is that when the model is training it gets stronger at classifying the classes, but when unseen data comes, the model doesn´t handle it well.

![image](https://github.com/user-attachments/assets/cc17494d-a332-4a33-b498-bec7d98781ba)

The training loss drops from 0.68% to 0.35%, gaining confidence with predictions on the training data, but in a similar way to the accuracy, the validation loss isn´t reduced as much, it even bounces a bit. The interpretation is that the model is starting to memorize instead of generalize.

![image](https://github.com/user-attachments/assets/3018e2c1-d46d-42c6-9739-1af55478a0d8)

The results in the confusion matrix show that the model tends to guess more the class Jerry, making safer guesses instead of actually guessing the truth.

The following indicators were also taken: 
* Precision (0.481)
* Recall (0.356)
* F1-score (0.409)

The precision and recall are low, meaning the model fails to generalize the distinction between Tom and Jerry especially missing Tom frequently, making it underfitting.

# Improved Model
The focus on the improvement was to increase the model´s precision. It was opted to use an architecture that had already proven effective on a similar classification task. In the paper written by Jian Xiao.(2020), the author uses a pre-trained VGG-based model to classify images of whether people are wearing a mask or not.

These were the main changes made:
* Switched from VGG16 to VGG19: It uses the same pre-trained weights from ImageNet but with more layers included.
* Changed the class_mode in the data generators(train_generator, test_generator, validation_generator): Since VGG19 was originally trained on multiple classes, the theory is that using class_mode='categorical' instead of 'binary' helps the model better understand the classification task by explicitly treating it as a multi-class problem, even with only two classes.
* Flatten and Dense layers: In the paper written by Jian Xiao. (2020), he proposes to replace the three fully connected layers of VGG19 with one Flatten layer and two dense layers for classification. In addition, the final Dense layer is changed from the binary activation 'Sigmoid' to the multiple class activation 'Softmax', to output probabilities for each class: [Tom, Jerry]

According to the paper, this adjustment improved the precision metric, increasing it from 86.71% to 97.62% when detecting people with a mask. So the idea is to boost the precision with the changes made.

![image](https://github.com/user-attachments/assets/199ba886-e4fb-408d-9b05-657c81741d1a)

# Results 
![image](https://github.com/user-attachments/assets/4c3ba61b-b3bb-4c66-9a19-6206e6fd237e)

The training accuracy starts low at 53%, and ends at 84% signaling that the architecture is slowly learning to capture patterns in the data. The validation accuracy starts at 61% and peaks at 72% but then goes down and stays between 64% and 69%, the consistency is limited but a bit better than the original model.

![image](https://github.com/user-attachments/assets/9f06fa8a-19ab-4e0a-bc27-e484774f6659)

The training loss goes down from 74% to 38%, and the validation loss also starts go down but around epoch 5 and 6 it starts to rise again. 

![image](https://github.com/user-attachments/assets/feb0c0be-6c81-4c10-980d-780886218ef9)

The following indicators were also taken: 
* Precision (0.491)
* Recall (0.439)
* F1-score (0.463)

Even with the more complex VGG19 model and the training improvements, the predictions still aren´t solid. The model is showing underfitting behavior again, meaning that the model can´t find strong enough patterns to distinguish the classes.

# Conclusion 
Despite implementing a well-documented improvement found in the paper "Application of a Novel and Improved VGG-19 Network in the Detection of Workers Wearing Masks", the model did not achieve stable results.
The dataset used, while functional, it introduced significant noise like images that contained multiple characters and varied backgrounds, all of which likely confused the model. This highlights how critical high-quality, well-curated datasets are, especially in tasks where visual differences can be subtle or context-dependent.

# References
B. Balabaskar, “Tom and Jerry Image Classification,” Kaggle, 2021. [Online]. Available: https://www.kaggle.com/datasets/balabaskar/tom-and-jerry-image-classification.
J. Xiao, J. Wang, S. Cao y B. Li, “Application of a novel and improved VGG-19 network in the detection of workers wearing masks,” Journal of Physics: Conference Series, vol. 1518, no. 1, p. 012041, 2020. doi: 10.1088/1742-6596/1518/1/012041.

