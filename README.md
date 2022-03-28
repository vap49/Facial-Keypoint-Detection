# CS301-Project Proposal
[Link to Kaggle Competition](https://www.kaggle.com/c/facial-keypoints-detection)
## Problem To Investigate
* In this project, we will be attempting to design a model that can predict the position of key facial features in humans ( such as eyes, nose, ears etc )
* Facial Recognition, while effortlessly performed by humans, is an incredibly challenging endeavor for machines. Because this task is easy for humans, it may seem trivial at first but it in actuality the amount of facial features and their variability/uniqueness from face to face is our main obstacle. Facial Recognition can be considered a problem of identifying an individual , or their feaures via images of said individual. Tasks such as head pose estimation for driver assist systems and government identification systems, facial tracking, facial signs for medical diagnosis, amongst others, rely on accurate detection of facial keypoints in their operation.
## Readings
* For this project, we will examine the following readings
  * Facial Keypoint Detection with Convolutional Neural Networks by Savina Colaco and Dong Seog Han PhD
  * Facial Keypoint Detection by Shayne Longpre and Ajay Sohmshetty at the University Of Stanford
## Data
* The dataset chosen will be the Kaggle competition-facial keypoints detection
* The input images are given as 96x96 pixel images represented as a list of pixels ordered by row ranging from (0,255)
 #### Data File Contents
* The testing and training data is separated as such...
  * training.csv: contains a list of 7049 training images with rows containing cartesian coordinates for the 15 facial keypoints and image data, as explained above.
  * test.csv: 1783 images to test the model.
  * submissionFileFormat.csv: contains all of the keypoints present in each input image. "Location" is what our model will have to predict.
## Methodology
* To complete the facial keypoint recognition task, we will deploy a Convolutional Neural Network to aid in classifying each of the facial keypoints present in our images. The CNN will process the raw pixels of image input into multiple levels of feature representations and allow us to detect the features of our inputs to produce predictions. 
## Evaluation of Results
* With the model fully trained, I expect to see dots that represent the cartesian coordinates where the model predicts each facial feature to be present for each image.
