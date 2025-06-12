# Cartoon_Binary_Classification_Model

*José Eduardo Díaz Maldonado - A01735676*

# Abstract

This work presents a binary classification model using deep learning to distinguish between images of the cartoon characters Tom and Jerry. The projects explore different approaches such as transfer learning with VGG16 and VGG pre-trained models, the implementation of dense and convolutional layers, and multiple model refinements. The results suggest that while pre-trained models provide a solid foundation, the dataset characteristics play a fundamental role in the performance of the model.

# Introduction
Artificial Intelligence is transforming every industry by automating tasks and enhancing decision-making processes. However, AI is just the surface layer, underneath it lies Machine Learning, within it is Supervised Learning where models are trained with labeled data, deeper we encounter Neural Networks and within them are Deep Learning models that can manage complex data patterns. A powerful type of Deep Learning is Convolutional Neural Networks which excel in image tasks. This project aims to demonstrate how supervised learning is applied in the binary classification of images and with the goal to distinguish between two visual categories from a public dataset. Beyond model construction, this work aims to analyze and interpret the performance using metrics and indicators such as a precision, recall, and confusion matrix.

# Dataset Generation and Selection 
Initially, a custom dataset was built for a different project involving Saiyan transformations from the Dragon Ball series, but the dataset was eventually discarded due to limited size and variety. A larger binary image classification dataset was sourced from Kaggle, which focuses on the characters from Tom and Jerry.  The dataset is organized into three directories: train, validation and test, each with two categories.

| Class  | Train images | Validation Images | Test images |
| ------------- | ------------- | ------------- | ------------- | 
| tom | 800 | 180 | 180 
| jerry  | 800 | 180 | 180

Givin us a total of 1680 train images, 360 validation images and 360 test images.

# Data preprocessing
For model robustness, the training and validation images were preprocessed using TensorFlow´s *ImageDataGenerator*. The steps are the following:

* Pixel normalization: rescale = 1./255 to normalize pixel values between 0 and 1.
* Rotation augmentation: rotation_range = 240° to create visual variability by moving the image without losing key features.

All images were resized to (100, 100) before being passed to the model.

# Examples of data augmentation 
![image](https://github.com/user-attachments/assets/804ef9b7-d793-4876-bd93-f835aa26517e)

# Model Implementation
The first model used transfer learning with the VGG16 architecture, a convolutional neural network designed for image recognition tasks. Consists of 13 convolutional layers followed by 3 fully connected layers, using small 3*3 filters and max pooling to reduce spatial dimensions.

![image](https://github.com/user-attachments/assets/62169893-9deb-4e16-944a-a0d265ffbbc5)

The top classification layers were excluded with (include_top=False) because the original model was designed to classify 1000 classes from the ImageNet dataset. Additionally, the trainable qualities of VGG16 are stopped with (trainable = False). We only take advantage of the visual features it already knows how to detect. Plus, a custom set of dense layers was appended with 256 filters, culminating with a sigmoid output layer. 

| Layer (type)  | Output Shape | Param # | 
| ------------- | ------------- | ------------- | 
| vgg16 (Functional) | (None, 3, 3, 512) | 14,714,688 |
| flatten (Flatten) | (None, 4608) | 0 |
| dense1 (Dense) | (None, 256) | 1, 179, 904 |
| output (Dense) | (None, 1) | 257 |

The model is compiled with the RMSprop optimizer and binary_crossentropy as the loss function. It is then trained for 10 epochs.

# Initial Evaluation of the Model

![image](https://github.com/user-attachments/assets/0bf6a70d-8869-4ab2-84b9-662ae829c982)

Training accuracy began at 57.8% and steadily increased, reaching 87.3% by the tenth epoch, demonstrating that model was effectively learning patterns from the training data. Validation accuracy started higher at 62.5% but showed much slower growth, settling around 71.1%. The training improvement shows the model learnt effectively but, the limited growth in validation accuracy suggest difficulty in generalization, pointing to overfitting.

![image](https://github.com/user-attachments/assets/cc17494d-a332-4a33-b498-bec7d98781ba)

Training loss drops from 0.68% to 0.35%, gaining confidence with predictions on the training data, but in a similar way to accuracy, validation loss showed less reduction, even bouncing up in some results, indicating memorization over generalization.

![image](https://github.com/user-attachments/assets/3018e2c1-d46d-42c6-9739-1af55478a0d8)

The confusion matrix shows that the model struggled, often missing the class ‘Tom’, other indicators captured were the precision: 0481, recall: 0.356 and F1-score: 0.409, the low precision and recall, indicates that the model fails to generalize the distinction between the two classes, making safer guesses instead of actually guessing the truth, showing signs of underfitting.

# Second Model 

For the second model, it was opted to use an architecture that had already proven effective on a similar classification task. In the paper written by Jian Xiao´s (2020), the author uses the VGG19 model, a deeper version of VGG16 that uses the same pre-trained weights from ImageNet but with more layers. Following Jian Xiao´s (2020) paper, several changes were made in the model:

-	The class_mode in the data generators was changed from ‘binary’ to ‘categorical’. This change was based on the idea that VGG19 was originally trained on a multi-class dataset, and framing the task in a similar multi-class context (even with only two classes) could improve its performance. 
-	The final classification layers were modified. Instead of using the default fully connected layers of VGG19, a Flatten layer was followed by two custom dense layers, and a final output layer that uses softmax activation to provide probabilities for both class [Tom, Jerry]. 

![image](https://github.com/user-attachments/assets/fa6c2518-8ad7-42fc-b112-5108d894df40)

In Jian Xiao´s study, these architectural changes increased precision significantly. Similarly, this approach was taken to improve the model´s precision.

| Layer (type)  | Output Shape | Param # | 
| ------------- | ------------- | ------------- | 
| vgg19 (Functional) | (None, 3, 3, 512) | 20,024,384 |
| flatten_2 (Flatten) | (None, 4608) | 0 |
| dense_layer1 (Dense) | (None, 512) | 2, 359, 808 |
| dense_layer2 (Dense) | (None, 256) | 131, 328 |
| output (Dense) | (None, 2) | 514 |

# Initial Evaluation of the second model

![image](https://github.com/user-attachments/assets/07c58d2a-9a6d-490b-9f36-66722d1b813a)
![image](https://github.com/user-attachments/assets/2ee1ce2e-c11a-405c-b5cc-ad1630a3e7bf)

The training accuracy starts low at 62%, and ends at 85% signaling that the architecture is slowly learning to capsule patterns in the data. The validation accuracy starts at 67% and peaks at 70% but then goes down. The training loss goes down from 64% to 37%, and the validation loss also starts to go down but around epoch 5 and 6 it starts to rise again.

![image](https://github.com/user-attachments/assets/62ec3993-2762-4391-93a0-a8cad5cdbe4b)

The confusion matrix shows that even with the improved model it still struggled, now missing the class ‘Jerry’, other indicators captured were the precision: 0491, recall: 0.594 and F1-score: 0.538. Even with the more complex model and training improvements, the predictions weren´t solid, the model is still showing underfitting behavior and the model can´t find strong enough patterns to distinguish the classes.

# Model Refinement 

Following a review with a mentor, It was determined that using softmax and categorical mode in a binary classification context was not optimal. While softmax is ideal for multiclass problems by assigning probabilities to each class, sigmoid is ideal for binary classification. Therefore, the model was redesigned using VGG19 but with binary architecture, employing a single Dense output layer with sigmoid activation. Dropout layers were introduced between dense layers to prevent overlearning by disabling neurons during training.

| Layer (type)  | Output Shape | Param # | 
| ------------- | ------------- | ------------- | 
| vgg19 (Functional) | (None, 3, 3, 512) | 20,024,384 |
| flatten_2 (Flatten) | (None, 4608) | 0 |
| dense_layer3 (Dense) | (None, 512) | 2, 359, 808 |
| dropout_3 (Dropout) | (None, 512) | 0 |
| dense_layer4 (Dense) | (None, 256) | 131, 328 |
| dropout_4 (Dropout) | (None, 512) | 0 |
| dense_layer5 (Dense) | (None, 256) | 65, 792 |
| dropout_5 (Dropout) | (None, 256) | 0 |
| output (Dense) | (None, 1) | 257 |

To further evaluate the impact of transfer learning, a second model was put to the test without relying on any pretrained base. This architecture was built entirely from scratch, using a sequential arrangement of convolutional, pooling, and dropout layers designed to extract features progressively. The goal was to benchmark performance against the VGG19-based model and determine whether feature extraction from a pretrained network provided significant advantages in accuracy and generalization. By training both models under the same conditions, a fair comparison could be drawn regarding training time, convergence behavior, and validation metrics.

| Layer (type)                        | Output Shape         | Param #     |
|------------------------------------|-----------------------|-------------|
| conv2D_20 (Conv2D)                 | (None, 98, 98, 200)   | 5,600       |
| max_pooling2d_20 (MaxPooling2D)   | (None, 49, 49, 200)   | 0           |
| dropout_18 (Dropout)              | (None, 49, 49, 200)   | 0           |
| conv2D_21 (Conv2D)                 | (None, 47, 47, 400)   | 720,400     |
| max_pooling2d_21 (MaxPooling2D)   | (None, 23, 23, 400)   | 0           |
| dropout_19 (Dropout)              | (None, 23, 23, 400)   | 0           |
| conv2D_22 (Conv2D)                 | (None, 21, 21, 600)   | 2,160,600   |
| max_pooling2d_22 (MaxPooling2D)   | (None, 10, 10, 600)   | 0           |
| dropout_20 (Dropout)              | (None, 10, 10, 600)   | 0           |
| conv2D_23 (Conv2D)                 | (None, 8, 8, 800)     | 4,320,800   |
| max_pooling2d_23 (MaxPooling2D)   | (None, 4, 4, 800)     | 0           |
| dropout_21 (Dropout)              | (None, 4, 4, 800)     | 0           |
| conv2D_24 (Conv2D)                 | (None, 2, 2, 1000)    | 7,201,000   |
| max_pooling2d_24 (MaxPooling2D)   | (None, 1, 1, 1000)    | 0           |
| dropout_22 (Dropout)              | (None, 1, 1, 1000)    | 0           |
| flatten_9 (Flatten)               | (None, 1000)          | 0           |
| dense_12 (Dense)                  | (None, 512)           | 512,512     |
| dropout_23 (Dropout)              | (None, 512)           | 0           |
| dense_13 (Dense)                  | (None, 1)             | 513         |

Additionaly, 104 images were manually added to each class in the training dataset to provide more learning for the model. The first model was trained for 30 epochs and the second one was shortened to 20 epochs.

- Confusion Matrix (VGG19 pretrained model)
  
![image](https://github.com/user-attachments/assets/fe7735f2-8e12-4ffb-b2d6-a79f8ae784c9)

- Confusion Matrix (Custom CNN)
  
![image](https://github.com/user-attachments/assets/49de5242-784c-4ca8-acd6-ba53b63a750f)

- Table of indicators from both models
  
| Metric       | VGG19 Pretrained Model | Custom CNN (from Scratch) |
|--------------|------------------------|----------------------------|
| Precision    | 0.508                  | 0.528                      |
| Recall       | 0.556                  | 0.528                      |
| F1-Score     | 0.531                  | 0.528                      |

Despite architectural complexity, the difference in performace between models is minimal.

# Conclusion 
Despite various implementations, from taking well-documented architecture from a paper, made from scratch architecture, and altering parameters, the model didn´t achieve stable results and couldn´t handle the classification with precision.

In this project, the dataset though functional presented several limitations that likely hindered the performance all the iterations of the model. Many images included multiple characters, complex or cluttered backgrounds. Such variability introduces label ambiguity and visual noise, which can confuse convolutional layers during feature extraction. For binary classification tasks where the difference between classes may be subtle or context-dependent, these factors become especially problematic. This evaluation highlights a fundamental principle in machine learning: a high-quality, well-curated dataset is often more valuable than advanced model architecture. 

# References
B. Balabaskar, “Tom and Jerry Image Classification,” Kaggle, 2021. [Online]. Available: https://www.kaggle.com/datasets/balabaskar/tom-and-jerry-image-classification.
J. Xiao, J. Wang, S. Cao y B. Li, “Application of a novel and improved VGG-19 network in the detection of workers wearing masks,” Journal of Physics: Conference Series, vol. 1518, no. 1, p. 012041, 2020. doi: 10.1088/1742-6596/1518/1/012041.

