/*
Title: Deep Learning in Computer Vision and its Applications
Decription: Computer Vision ML, DL Applications
Author: Bhaskar Mangal
Date: 30 Jun 2018
Last Updated: 12 Jul 2018
Tags: Computer Vision ML, DL Applications
*/


**Table of Contents**

[TOC]

# Deep Learning in Computer Vision and its Applications
> Provide APIs in Computer Vision using Deep Learning for Geospatial Industry

* [AI Programme Slides](https://github.com/mangalbhaskar/dia/blob/master/AI-programme-slides.pdf)
* [Select / Building Deep Learning Machine for Computer Vision](hardwares-configs.md)
* [Deep Learning Frameworks](deep-learning-frameworks.md)
* Installation and System Setup
  * [System Setup - start here](https://github.com/mangalbhaskar/linuxscripts/blob/master/README.md#start-here---the-big-bang-theory)
  * [Installing Deep Learning Frameworks on Ubuntu with CUDA Support](https://www.learnopencv.com/installing-deep-learning-frameworks-on-ubuntu-with-cuda-support/)
* [Datasets and Data Creation for Training Machines](deep-learning-datasets-and-creation.md)
* [Getting Started for Image Processing, Computer Vision, Machine Learning and Deep Learning for GIS](https://github.com/mangalbhaskar/technotes#getting-started-for-image-processing-computer-vision-machine-learning-and-deep-learning-for-gis)
* [Deep Learning Concepts](deep-learning.md)
* [Mechanics of Self Driving Car](mechanics-of-self-driving-car.md)

## Problem Statements

### 1. **Signage Layer Extraction for Map**

* a. Trees, Signage, Road Signs,Traffic Signals, Poles
* b. In combination with photogrammetry (point clouds  to get higher accurate geolocation

### 2. Number Plate Recognition

**Potential use cases:**
* a. In traffic camera feeds for traffic rules violation detection
* b. Toll both automation

###	3. Vehicle Detection & Recognition

**Potential use cases:**
* a. In traffic camera feeds
	­ - Vehicle counts
	­ - Traffic density estimation
* b. Parking lots vacancy detection
* c. Accident / collision detection

### 4. Pedestrian Detection & Face Recognition

**Potential use cases:**
* a. Surveillance & security like ATM, Malls, Real Estate

### 5. Road Profile Extraction for Map

**Potential use cases:**
* a. Edge detection ­ road, footpath, road markings
* b. In combination with photogrammetry (point clouds) to get higher accurate geolocation

### 6. **Cross Road / Junction Identification**

### 7. **Text Extraction for Signage Layer**

### 8. **Complete Urban scene Classification & Segmentation**

### 9. **Content retrieval based on above**

### 10. **Other problem statements**

- Face Detection
- Removing motion blur
- Assisted Driving
  * Pedistrian and Car Detection
  * Lane Detection
    - Collision warning systems with adaptive cruise control
    - Lane departure warning systems
    - Rear object detection systems
- Iris recogni@on
- Visually defined search
- Object search in video
- Visual descrip@on – visual words
- Image representa@on using visual words
- Organizing photo collections
- Visual dictionary
- computer vision, natural language, and recommendation systems
- [AI ML DL Problem Statements](ai-ml-dl-problem-statements.md)
- [Deep Learning Applications](deep-learning-applications.md)

## Concepts

### **Image processing, Computer Vision**
* [Computer Vision](computer-vision.md)
* [Image Processing](image-processing.md)
* [OpenCV with Python](opencv-with-python.md)
* [Photogrammetry](photogrammetry.md)
* [visualSfM](visualSfM.md)
* [SLAM](slam.md)
* [Image based Navigation](image-based-navigation.md)

### **Machine Learning**
* [Machine Learning](machine-learning.md)
* [ML for Mobile and Web Applications](ml-for-mobile-and-web-applications.md)
* [CBIR](cbir.ml.md)

### **Mathematics, Statistics and Statistical Learning**
* [Statistics](stats.md)
* [Introduction to Statistical Learning Using R - ISLR (Book notes)](islr-book-notes.md)
* [Mathematics](maths.md)

### **Data Analytics and Visualization**
* [D3.js](d3.js.md)
* [Data Analytics](data-analytics.md)
* [Data Visualization in Web](data-visualization-in-web.md)

## Case Studies Specific to Problem Statements

### Traffic Light
* https://medium.freecodecamp.org/recognizing-traffic-lights-with-deep-learning-23dae23287cc
  - https://github.com/davidbrai/deep-learning-traffic-lights
  - https://challenge.getnexar.com/challenge-1
  - https://arxiv.org/abs/1602.07360

### Traffic Sign
* https://arxiv.org/pdf/1712.04391.pdf
* https://hackernoon.com/traffic-signs-classification-with-deep-learning-b0cb03e23efb
* http://www.deeplearningbook.org/
* https://towardsdatascience.com/recognizing-traffic-signs-with-over-98-accuracy-using-deep-learning-86737aedc2ab
* https://github.com/chandansaha2014/Real-time-Traffic-Sign-Recognition
* https://becominghuman.ai/build-a-neural-network-based-traffic-sign-classification-system-with-98-5-ed42a9273a20
* https://ip.cadence.com/uploads/901/cnn_wp-pdf
* https://mc.ai/resnet-for-traffic-sign-classification-with-pytorch/

**Traffic Sign Recognition with TensorFlow**
* https://medium.com/@waleedka/traffic-sign-recognition-with-tensorflow-629dffc391a6
* https://github.com/waleedka/traffic-signs-tensorflow
* https://mc.ai/the-traffic-sign-classifier-project/
* http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf
* https://pypi.org/project/pytesseract/

**Details**
* Detect and recognize iconic symbol and text strings contained in road panels
* Different types of traffic signs as Information signs, Warning signs, Mandatory signs and prohibited signs are used and these signs helps driver to achieve efficient navigation and safe driving.
* The major problem are changing lighting conditions in outdoor environments, the obstructions of objects between the cameras and the traffic panels [5], partially damaged traffic signs, long exposure of sunlights lead to faded color. 
* computer vision technique is carried out frequently The road sign can be categorized to three properties by color (blue, red, green and brown), shape (circular, square, triangular and octagonal) and the inner part of the sign [4], which plays a major role in the detection stage in traffic sign detection and recognisation.
* The speed and efficiency of the detection and recognisation of the traffic sign plays the important role in the system.

**Techniques in detection and recognisation of Traffic Signs**
- overview of the system
- Detection stage used to extract the traffic sign based on the
  - hape and color features,
- Recognisaton stage to classify the traffic sign

**Concludes**
* It is normally based on color or shape segmentation algorithms
  - The color segmentation is usually a binary mask to separate the interested target objects from the background.
  - The region of interest is determined by the connected components
  - The shape features are extracted in the binary image to detect the sign by verifying the hypothesis of the sign. The recognisation stage determines the type of traffic sign 

**Goal**
1. Classify traffic signs using a simple convolutional neural network.

**Data Exploration**
- Most image classification networks expect images of a fixed size
- But since the images have different aspect ratios, then some of them will be stretched vertically or horizontally. Is that a problem? I think it’s not in this case, because the differences in aspect ratios are not that large. My own criteria is that if a person can recognize the images when they’re stretched then the model should be able to do so as well.
- Some stats about the data
  * How many images and labels available?
  * How many unique labels are there?
  * What are the sizes of the images?
  * What is the aspect ratio for each image - min, max, median, standard deviation? Is this aspect ratio variation is too huge to cause distortion such that images are not even recognizable by human themself?
  * How the aspect ratio, number of images varies within each label group? Illustrate with the graphs and plot.

**Developing Intution**
1. Get the stats on width, height and Aspect ratio of the dataset
  - In early development, use a smaller size because it leads to faster training, which allows me to iterate faster
  - Generally, 16x16 and 20x20, are too small.
  - 28x28 and 32x32 are better choice and which is easy to recognize
2. Verify the range of the data and catch bugs early
  - Printing the `min()` and `max()` values. This tells that the image colors are the standard range of `0–255`

**Start Simple**
- Start with the simplest possible model: A one layer network that consists of one neuron per label.

**Tips**
- do Exploratory Data analysis. Knowing data well from the start saves a lot of time later.
- Pre-process and Handling Images of Different Sizes
- In early development use a smaller size because it leads to faster training, which allows to iterate faster
- Experiment with 16x16 and 20x20, but if they were too small that pick 32x32 which is easy to recognize and reduces the size of the model and training data by a factor of 16 compared to 128x128
- Get and print the min() and max() values. It’s a simple way to verify the range of the data and catch bugs early.
- Activation function: ReLU, sigmoid, tanh, fully connected layer, logits vector
- convert logits to probabilities using the softmax function if needed. If not needed, get the index of the largest value, which corresponds to the id of the label. The `argmax` op does that.
  - https://www.quora.com/What-are-the-benefits-of-using-rectified-linear-units-vs-the-typical-sigmoid-activation-function
  - http://cs231n.github.io/neural-networks-1/
- `cross-entropy` is the most common function for classification tasks.
- **Cross-entropy is a measure of difference between two vectors of probabilities**
  - https://jamesmccaffrey.wordpress.com/2013/11/05/why-you-should-use-cross-entropy-error-instead-of-classification-error-or-mean-squared-error-for-neural-network-classifier-training/
  - http://neuralnetworksanddeeplearning.com/chap3.html#the_cross-entropy_cost_function
- Convert labels and the logits to probability vectors. The function `sparse_softmax_cross_entropy_with_logits()` simplifies that
  - It takes the input as generated logits and the groundtruth labels
  - It does three things:
    * converts the label indexes of shape [None] to logits of shape [None, 62] (one-hot vectors)
    * then it runs softmax to convert both prediction logits and label logits to probabilities
    * and finally calculates the cross-entropy between the two
- This generates a loss vector of shape `[None]` (1D of length = batch size), which we pass through `reduce_mean()` to get one single number that represents the loss value.
```bash
loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels_ph)
      )
```
- Choosing the optimization algorithm is another decision to make.
  - ADAM optimizer has shown to converge faster than simple gradient descent
  - http://sebastianruder.com/optimizing-gradient-descent/index.html
```bash
train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
```
- The last node in the graph is the initialization op, which simply sets the values of all variables to zeros (or to random values or whatever the variables are set to initialize to).
```bash
init = tf.initialize_all_variables()
```
- Notice that the code above doesn’t execute any of the ops yet. It’s just building the graph and describing its inputs. The variables we defined above, such as init, loss, predicted_labels don’t contain numerical values. They are references to ops that we’ll execute next.
- "Training Loop" is where we iteratively train the model to minimize the loss function. Before we start training, though, we need to create a **Session** object

**Pre-processing**
**Handling Images of Different Sizes**
- Most image classification networks expect images of a fixed size. So we need to resize all the images to the same size.
- our first model will do as well.
- If the images have different aspect ratios, then some of them will be stretched vertically or horizontally.
  - Is that a problem? May not be in some cases, because when the differences in aspect ratios are not that large and a person can recognize the images when they’re stretched then the model should be able to do so as well.

**Minimum Viable Model**
- start with the simplest possible model
- A one layer network that consists of one neuron per label.
- This network has `62` neurons and each neuron takes the RGB values of all pixels as input.
- Effectively, each neuron receives `32*32*3=3072` inputs. This is a fully-connected layer because every neuron connects to every input value
- Once this works end to end, expanding on it is much easier than building something complex from the start.

**Building the TensorFlow Graph**
- TensorFlow encapsulates the architecture of a neural network in an execution graph.
- The graph consists of operations (Ops for short) such as Add, Multiply, Reshape, …etc.
- These ops perform actions on data in tensors (multidimensional arrays).
- First, Graph object. TensorFlow has a default global graph, but I don’t recommend using it. Global variables are bad in general because they make it too easy to introduce bugs. Hence, create the graph explicitly.
```
graph = tf.Graph()
```
- define Placeholders for the images and labels. The placeholders are TensorFlow’s way of receiving input from the main program
- The shape of the images_ph placeholder is `[None, 32, 32, 3]`. It stands for [batch size, height, width, channels] (often shortened as NHWC) . The None for batch size means that the batch size is flexible, which means that we can feed different batch sizes to the model without having to change the code.
- Pay attention to the order of your inputs because some models and frameworks might use a different arrangement, such as NCHW.
- If layer expects input as a one-dimensional vector, flatten the images first.
```
# Flatten input from: [None, height, width, channels]
# To: [None, height * width * channels] == [None, 3072]
images_flat = tf.contrib.layers.flatten(images_ph)
```

**Loss Function and Gradient Descent**

#### Potential Applications
- Intelligent Transportation Systems (ITS)
- Traffic Surveillance System
- ADAS

#### Corner cases - Potential pitfalls
1. Road signs with text written
- for example, a Road sign for 'No Entry for HTV', but an exception of School Buses. This type exists at 13th Main Road, Indranagar next to MMI office in bangalore.

#### Paper Reviews
1. TRAFFIC-SIGN RECOGNITION FOR AN INTELLIGENT VEHICLE/DRIVER ASSISTANT SYSTEM USING HOG 
- http://aircconline.com/cseij/V6N1/6116cseij02.pdf
- To recognize the traffic sign, the system has been proposed with three phases. They are Traffic board Detection, Feature extraction and Recognition. The detection phase consists of RGBbased colour thresholding and shape analysis, which offers robustness to differences in lighting situations. A Histogram of Oriented Gradients (HOG) technique was adopted to extract the features from the segmented output. Finally, traffic signs recognition is done by k-Nearest Neighbors (k-NN) classifiers. It achieves an classification accuracy upto 63%. 

2. AUTOMATED TRAFFIC SIGN BOARD CLASSIFICATION SYSTEM
- https://wireilla.com/papers/ijcsa/V5N1/5115ijcsa06.pdf
- Intelligent sign board classification method based on blob analysis in traffic surveillance
- A Sign board is modelled as a rectangular patch and classified via blob analysis. By processing the blob of sign boards, the meaningful features are extracted. Tracking moving targets is achieved by comparing the extracted features with training data. After classifying the sign boards the system will intimate to user in the form of alarms, sound waves. The experimental results show that the proposed system can provide real-time and useful information for traffic surveillance. 

3. SURVEY-AN EXPLORATION OF VARIOUS TECHNIQUES FOR SIGN DETECTION IN TRAFFIC PANELS
- http://www.arpnjournals.com/jeas/research_papers/rp_2015/jeas_0515_2045.pdf
- a survey of the traffic sign detection and recognition, to detail the system for driver assistance to ensure safe journey

4. Indian Traffic Sign Detection and Classification Using Neural Networks
- http://ijoes.vidyapublications.com/paper/Vol19/03-Vol19.pdf

5. ROAD AND TRAFFIC SIGN DETECTION AND RECOGNITION 
- http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.104.2523&rep=rep1&type=pdf

6. Real Time Detection and Recognition of Indian Traffic Signs using Matlab
- https://www.ijser.org/researchpaper/real-time-detection-and-recognition-of-indian-traffic-signs-using-matlab.pdf

### Face Detection
- https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/
- https://www.pyimagesearch.com/2018/07/09/face-clustering-with-python/

**Face recognition with OpenCV, Python, and deep learning**
* with deep learning you know that we typically train a network to:
  - Accept a single input image
  - And output a classification/label for that image
* deep metric learning
  - Instead, of trying to output a single label (or even the coordinates/bounding box of objects in an image), we are instead outputting a real-valued feature vector
* For the dlib facial recognition network, the output feature vector is 128-d (i.e., a list of 128 real-valued numbers) that is used to quantify the face. Training the network is done using triplets
*  Facial recognition via deep metric learning involves a “triplet training step.” The triplet consists of 3 unique face images — 2 of the 3 are the same person. The NN generates a 128-d vector for each of the 3 face images. For the 2 face images of the same person, we tweak the neural network weights to make the vector closer via distance metric.
* Here we provide three images to the network:
  - Two of these images are example faces of the same person.
  - The third image is a random face from our dataset and is not the same person as the other two images.
* Our network architecture for face recognition is based on **ResNet-34** from the **Deep Residual Learning for Image Recognition** paper by He et al., but with fewer layers and the number of filters reduced by half.
* The network itself was trained by Davis King on a dataset of ~3 million images. On the **Labeled Faces in the Wild (LFW)** dataset the network compares to other state-of-the-art methods, reaching 99.38% accuracy.
- Davis King (the creator of dlib)
- Adam Geitgey (the author of the face_recognition module we’ll be using shortly)
- [face_recognition](https://github.com/ageitgey/face_recognition)
- [dlib](http://dlib.net/)
```bash
sudo pip install dlib
# OR, compile from: git clone https://github.com/davisking/dlib.git # preferred for GPU, CUDA support
#
sudo pip install face_recognition
#pyimagesearch utility
sudo pip install imutils
```

#### OpenCV Face Detection
- https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/
- OpenCV ships out-of-the-box with pre-trained Haar cascades that can be used for face detection
- “hidden” deep learning-based face detector that has been part of OpenCV since OpenCV 3.3

**Objectives**
- How you can perform face detection in images using OpenCV and deep learning
- How you can perform face detection in video using OpenCV and deep learning

**DNN Face Detector**
- The Caffe-based face detector can be found in the face_detector sub-directory of the dnn samples:
- https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector
- When using OpenCV’s deep neural network module with Caffe models, you’ll need two sets of files:
  * The .prototxt file(s) which define the model architecture (i.e., the layers themselves)
  * The .caffemodel file which contains the weights for the actual layers
- OpenCV’s deep learning face detector is based on the Single Shot Detector (SSD) framework with a ResNet base network (unlike other OpenCV SSDs that you may have seen which typically use MobileNet as the base network).

### Semantic Segmentation
- https://github.com/tensorflow/models/tree/master/research/deeplab
- https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html

**Mapillary**
* https://github.com/lopuhin/mapillary-vistas-2017
* https://github.com/mapillary/mapillary_vistas
* https://arxiv.org/pdf/1803.05675.pdf
* http://cs231n.stanford.edu/reports/2017/pdfs/633.pdf

**Training on Mapillary dataset**
- https://oslandia.com/en/2018/05/07/deeposlandia-0-4-has-been-released/
- https://github.com/Oslandia/deeposlandia



- https://www.pyimagesearch.com/2018/07/19/opencv-tutorial-a-guide-to-learn-opencv/
- https://www.pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/

### Optical Mark Recognition (OMR)
* https://www.pyimagesearch.com/2016/10/03/bubble-sheet-multiple-choice-scanner-and-test-grader-using-omr-python-and-opencv/

**Possible use cases**
- Automatic Data digitigation of Field Survey Data, for example information collected of a point of interest and recorded in an OMR sheet
- Simple Hand written digit (single) recognition from the survey data


## Generative Networks
- https://github.com/hardmaru/cppn-tensorflow

## CPPNs - Compositional pattern-producing networks
- https://en.wikipedia.org/wiki/Compositional_pattern-producing_network

- Compositional pattern-producing networks (CPPNs) are a variation of artificial neural networks (ANNs) that have an architecture whose evolution is guided by genetic algorithms.
- The architect of a CPPN-based genetic art system can bias the types of patterns it generates by deciding the set of canonical functions to include.
- Since they are compositions of functions, CPPNs in effect encode images at infinite resolution and can be sampled for a particular display at whatever resolution is optimal.

**Usage of different canonical functions patterns:**
- periodic functions such as sine produce segmented patterns with repetitions,
- symmetric functions such as Gaussian produce symmetric patterns
- Linear functions can be employed to produce linear or fractal-like patterns.


## Neuroevolution
Neuroevolution, or neuro-evolution, is a form of artificial intelligence that uses evolutionary algorithms to generate artificial neural networks (ANN), parameters, topology and rules.
- It is most commonly applied in artificial life, general game playing[2] and evolutionary robotics.

* TWEANNs -  Topology and Weight Evolving Artificial Neural Networks

**References**
- https://en.wikipedia.org/wiki/Neuroevolution
- https://en.wikipedia.org/wiki/Memetic_algorithm

## Blogs
- http://www.wildml.com/
- https://www.theatlantic.com/technology/archive/2017/08/inside-waymos-secret-testing-and-simulation-facilities/537648/
- https://nips2017creativity.github.io/

## Research Papers
- https://arxiv.org/abs/1711.06396
  * https://arxiv.org/pdf/1711.06396.pdf

Number Plate Detection
- http://matthewearl.github.io/2016/05/06/cnn-anpr/
