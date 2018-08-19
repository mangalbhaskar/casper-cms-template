/*
Title: Deep Learning
Decription: Deep Learning
Author: Bhaskar Mangal
Date: 
Last updated: 24th-Jul-2018
Tags: Deep Learning
*/

**Table of Contents**

[TOC]

## Deep Learning Frameworks, Toolchain, Libraries
* [Refer: Deep Learning Frameworks, Toolchain, Libraries](deep-learning-frameworks.md)

## Datasets and Data Creation for Training Machines
* [Refer: Datasets and Data Creation for Deep Learning](deep-learning-datasets-and-creation.md)

## What do Machine Learning practitioners do?
* http://www.fast.ai/2016/12/08/org-structure/

The processes of appropriately framing a business problem, collecting and cleaning the data, building the model, implementing the result, and then monitoring for changes are interconnected in many ways that often make it hard to silo off just a single piece (without at least being aware of what the other pieces entail)

While it’s common to have machine learning, engineering, and data/pipeline/infrastructure engineering all as separate roles, try to avoid this as much as possible. This leads to a lot of duplicate or unused work, particularly when these roles are on separate teams. You want people who have some of all these skills: can build the pipelines for the data they need, create models with that data, and put those models in production.
You’re not going to be able to hire many people who can do all of this. So you’ll need to provide them with training. 

Tech companies waste their employees’ potential by not offering enough opportunities for on-the-job learning, training, and mentoring. Your people are smart and eager to learn. Be prepared to offer training, pair-programming, or seminars to help your data scientists fill in skills gaps. 

Even when you have people who are both data scientists and engineers (that is, they can create machine learning models and put those models into production), you still need to have them embedded in other teams and not cordoned off together. Otherwise, there won’t be enough institutional understanding and buy-in of what they’re doing, and their work won’t be as integrated as it needs to be with other systems.

**what do machine learning practitioners do?**
- Understanding the context:
- Data:
- Modeling:
- Productionize:
- Monitor:

### **AutoML and Neural Architecture Search**
- As it’s name suggests, AutoML is one field in particular that has focused on automating machine learning, and a subfield of AutoML called neural architecture search is currently receiving a ton of attention.
- https://www.youtube.com/watch?v=kSa3UObNS6o
- The term AutoML has traditionally been used to describe automated methods for model selection and/or hyperparameter optimization.
- https://www.automl.org/automl/
- AutoML provides a way to select models and optimize hyper-parameters. It can also be useful in getting a baseline to know what level of performance is possible for a problem.
- What Pichai refers to as using **“neural nets to design neural nets”** is known as **neural architecture search**; typically **reinforcement learning** or **evolutionary algorithms** are used to design the **new neural net architectures**.
  - NASNet:1800 GPU days (the equivalent of almost 5 years for 1 GPU) 
  - AmoebaNet: 3150 GPU days (the equivalent of almost 9 years for 1 GPU)
  - Efficient Neural Architecture Search (ENAS): 16hrs, 1 GPU
  - DARTS - Differentiable architecture search (DARTS)
    - To learn a network for Cifar-10, DARTS takes just 4 GPU days, compared to 1800 GPU days for NASNet and 3150 GPU days for AmoebaNet (all learned to the same accuracy). This is a huge gain in efficiency! 
- how can humans and computers work together to make machine learning more effective?

## FAQs - Technical Questions

* **How many nodes to have in the hidden layer?**
	* The size of the input layer is fixed by the input space (e.g. one node per input pixel), and the size of the output layer is fixed by the output space (e.g. one node per category, in a classification task), but the inner hidden layer has no limitations on size. If the hidden layer is too small, it won't be able to separate out important "patterns" in the high-dimension space it works in. If the layer is too large, the relative contribution of each node will be small, and it will probably take longer to train. The ideal number of nodes depends on the problem the network is tasked with.
* **How to initialize the weights?**
	* Is it better to initialize all of the weights to 1, or initialize them randomly? Most sources seem to agree it's better to initialize the weights randomly, because this seems to decrease the probability that the optimization algorithm will get stuck in a bad local minimum. There are also recommendations to initialize the weights to small numbers (i.e. close to 1), so that no weight overpowers the others from the outset.
* **Stopping criteria?**
	* When do we stop training? Do we attempt to reach a certain error rate in the classification of the input set? Or do we iterate a specified number of times? It's probably best to give the algorithm a combination of different stopping criteria.
* **Learning rate?**
	* Many sources recommend damping the dqdq (for any node qq in the network) with a learning rate, which is usually somewhere around 0.3. This helps make sure that the gradient descent does not jump past a local minimum and miss it entirely.
* **How to feed in training data?**
	* Given a large set of input data, it may be tempting to train the network on a small portion of it until it excels there, and then gradually increase the data set size. This will not work: getting the network to excel on a small portion of the data will over-fit the network to that data. To learn new information afterward, the network will essentially have to forget what it learned before. A better way to train the network is to sample randomly (or in some uniform manner) from the overall dataset, so that the node never over-fits to its input.
---
1. https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/
2. http://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/
3. http://scs.ryerson.ca/~aharley/vis/conv/flat.html
4. http://scs.ryerson.ca/~aharley/neural-networks/
* some primer for matlab helpful to go through the NN code in the above link
	http://mathesaurus.sourceforge.net/matlab-numpy.html
	http://www.cert.org/flocon/2011/matlab-python-xref.pdf
	https://www.tutorialspoint.com/matlab/matlab_arrays.htm
	https://stackoverflow.com/questions/8625990/size-function-in-matlab
As you know, matlab deals mainly with matrices. So, the size function gives you the dimension of a matrix depending on how you use it. For example:
1. If you say size(A), it will give you a vector of size 2 of which the first entry is the number of rows in A and the second entry is the number of columns in A.
2. If you call size(A, 1), size will return a scalar equal to the number of rows in A.
3. If you call size(A, 2), size will return a scalar equal to the number of columns in A.

A scalar like scale in your example is considered as a vector of size 1 by 1. So, size(scale, 2) will return 1, I believe.

---

* How can I train a CNN with insufficient and not-so-good data?
* How do I classify images with non-rectangle shape with CNN?
* Why should we give an images with a fixed size for traditional CNN, whereas for FCN we can give an image with an arbitrary image size?
* Why do l need to apply PCA whitening to my images (black and white) before training my convolutional neural network (CNN)?
* What are the different types of input vectors used for an image in a neural network?
* How does the conversion of last layers of CNN from fully connected to fully convolutional allow it to process images of different size?
* How do I read image data for CNN in python?
	- https://www.quora.com/How-do-I-read-image-data-for-CNN-in-python
	- http://pillow.readthedocs.io/en/3.0.x/handbook/tutorial.html#using-the-image-class
* What-is-the-main-difference-between-a-Bayessian-neural-network-and-other-convolutional-neural-networks
	- A Bayesian network is a graphical model that works by modeling dependencies between nodes, by considering the conditional dependence (where zero implies independence). On the other hand Bayesian Convolutional Neural Networks are an adaptation of CNNs that helps to prevent overfitting and are therefore preferred in problems where CNNs are the appropriate DL model but there is insufficient data (and standard CNNs would therefore overfit on this small set of data as they very rapidly overfit). This is done by the introduction of uncertainity estimation in Bayesian Convolutional Neural Networks. There is an excellent paper explaining applications of Bayesian Convolutional Neural Nets by Gal and Ghahramani (2016).
	- CNN is a NN with convolutional and pooling layers which can randomly initialize (or uniform or xavier) its weights and biases whereas Bayesian NN takes prior distribution on its weights and biases.
	* https://www.quora.com/What-is-the-main-difference-between-a-Bayessian-neural-network-and-other-convolutional-neural-networks
	* https://en.wikipedia.org/wiki/Convolutional_neural_network
	* http://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/
When designing the architecture of a neural network you have to decide on:
- How do you arrange layers?
- Which layers to use?
- How many neurons to use in each layer etc.?
Once you have decided the architecture of the network;
- the second biggest variable is the weights(w) and biases(b) or the parameters of the network.
- What is the objective of the training?
The objective of the training is to get the best possible values of the all these parameters which solve the problem reliably. For example, when we are trying to build the classifier between dog and cat, we are looking to find parameters such that output layer gives out probability of dog as 1(or at least higher than cat) for all images of dogs and probability of cat as 1((or at least higher than dog) for all images of cats.

You can find the best set of parameters using a process called Backward propagation, i.e. you start with a random set of parameters and keep changing these weights such that for every training image we get the correct output. There are many optimizer methods to change the weights that are mathematically quick in finding the correct weights. GradientDescent is one such method(Backward propagation and optimizer methods to change the gradient is a very complicated topic. But we don’t need to worry about it now as Tensorflow takes care of it).
- Which architectures should I use? Should I create my own architecture to get started with?
	- There are many standard architectures which work great for many standard problems. Examples being AlexNet, GoogleNet, InceptionResnet, VGG etc. In the beginning, you should only use the standard network architectures. You could start designing networks after you get a lot of experience with neural nets. Hence, let’s not worry about it now.
**Machine Learning** and the role it plays in computer vision, image classification, and deep learning.
- http://neuralnetworksanddeeplearning.com/index.html
- http://neuralnetworksanddeeplearning.com/chap1.html
once we've learned a good set of weights and biases for a network, it can easily be ported to run in Javascript in a web browser, or as a native app on a mobile device. 

I had to make specific choices for the number of epochs of training, the mini-batch size, and the learning rate, ηη. As I mentioned above, these are known as hyper-parameters for our neural network, in order to distinguish them from the parameters (weights and biases) learnt by our learning algorithm.


But if we were coming to this problem for the first time then there wouldn't be much in the output to guide us on what to do. We might worry not only about the learning rate, but about every other aspect of our neural network. We might wonder if we've initialized the weights and biases in a way that makes it hard for the network to learn? Or maybe we don't have enough training data to get meaningful learning? Perhaps we haven't run for enough epochs? Or maybe it's impossible for a neural network with this architecture to learn to recognize handwritten digits? Maybe the learning rate is too low? Or, maybe, the learning rate is too high? When you're coming to a problem for the first time, you're not always sure.

we need to develop heuristics for choosing good hyper-parameters and a good architecture.

support vector machine or SVM
* http://peekaboo-vision.blogspot.in/2010/09/mnist-for-ever.html
* http://neuralnetworksanddeeplearning.com/chap2.html
	- algorithm for computing such gradients, an algorithm known as backpropagation.
  - how quickly the cost changes when we change the weights and biases.

**Interview Excerpt**
* https://www.pyimagesearch.com/2018/07/02/an-interview-with-francois-chollet/
People gravitate towards incremental architecture tricks that kinda seem to work if you don’t test them adversarially. They use weak baselines, they overfit to the validation set of their benchmarks. Few people do ablation studies (attempting to verify that your empirical results are actually linked to the idea you’re advancing), do rigorous validation of their models (instead of using the validation set as a training set for hyperparameters), or do significance testing.

We should remember that the purpose of research is to create knowledge. It’s not to get media coverage, nor is it to publish papers to get a promotion.

Mathematical notation can be a huge accessibility barrier, and it isn’t at all a requirement to understand deep learning clearly. Code can be in many cases a very intuitive medium to work with mathematical ideas.


## Research Papers
* https://github.com/ujjwalkarn/deeplearning-papernotes

## References
* http://lvdmaaten.github.io/tsne/
	- t-Distributed Stochastic Neighbor Embedding (t-SNE)
* http://cs231n.github.io/classification/
* http://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf
* http://people.csail.mit.edu/torralba/shortCourseRLOC/index.html
* https://www.coursera.org/learn/machine-learning
* http://cs231n.github.io/convolutional-networks/
* http://cs231n.stanford.edu/
* https://github.com/cberzan/highway-sfm
* https://www.cse.wustl.edu/~furukawa/
* https://github.com/mapillary/OpenSfM
* http://blog.mapillary.com/product/2017/05/03/mapillary-vistas-dataset.html
* http://blog.mapillary.com/update/2016/10/31/denser-3d-point-clouds-in-opensfm.html
* http://blog.mapillary.com/update/2016/09/27/semantic-segmentation-object-recognition.html
* http://openmvg.readthedocs.io/en/latest/software/SfM/IncrementalSfM/
* http://vhosts.eecs.umich.edu/vision//projects/ssfm/
* http://cvgl.stanford.edu/resources.html
* http://cvgl.stanford.edu/3d-r2n2/
* http://www.cs.cornell.edu/~asaxena/learningdepth/ijcv_monocular3dreconstruction.pdf
* http://www.theia-sfm.org/sfm.html

## Books
* https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/chapter1.html

Adaptive structure from motion with a contrario model estimation
Incremental SfM
[ACSfM]	Adaptive structure from motion with a contrario model estimation. Pierre Moulon, Pascal Monasse, and Renaud Marlet. In ACCV, 2012.

We consider the task of 3-d depth estimation from a single still image. We take a supervised learning approach to this problem, in which we begin by collecting a training set of monocular images (of unstructured indoor and outdoor environments which include forests, sidewalks, trees, buildings, etc.) and their corresponding ground-truth depthmaps. Then, we apply supervised learning to predict the value of the depthmap as a function of the image.

## MOOC Courses
* https://www.coursera.org/learn/neural-networks
* http://cs231n.github.io/
* http://deeplearning.net/tutorial/
* http://www.fast.ai/2017/09/08/introducing-pytorch-for-fastai/
* http://course.fast.ai/

## AL-ML-DL blogs
- http://www.fast.ai/2018/07/12/auto-ml-1/
- http://www.fast.ai/2018/04/30/dawnbench-fastai/
- https://www.forbes.com/sites/nvidia/2018/08/02/how-swiss-federal-railway-is-improving-passenger-safety-with-the-power-of-deep-learning/#10fc74a650e3
- https://www.linkedin.com/pulse/how-set-up-nvidia-gpu-enabled-deep-learning-development-tianyi-pan
- https://hackernoon.com/setting-up-your-gpu-machine-to-be-deep-learning-ready-96b61a7df278
- http://timdettmers.com/2014/09/21/how-to-build-and-use-a-multi-gpu-system-for-deep-learning/
- https://becominghuman.ai/setting-up-deep-learning-gpu-environment-5651564ff936

## Deep Learning, CCN Terms and Concepts
* https://github.com/ml4a/ml4a.github.io/blob/master/_chapters/neural_networks.md

* Neural netowrk
* feed-forward network
* recursive neural network
* gradient descent
* sigmoid function
* sigmoidal neuron
* learning rate
* delta rule
* stochastic gradient descent
* logit
* backpropogation
* overfiting
* validation
* cross-validation
* Bias trick
* center your data
* high-dimensional column vectors/points
	* https://www.khanacademy.org/math/linear-algebra/vectors-and-spaces/vectors/v/real-coordinate-spaces
* score function
* loss function (cost function or the objective)
	* The loss function quantifies our unhappiness with predictions on the training set
* SVM - Support Vector Machine
* Multiclass Support Vector Machine
	* The Multiclass Support Vector Machine "wants" the score of the correct class to be higher than all other scores by at least a margin of delta
	* regularization function is not a function of the data, it is only based on the weights. Including the regularization penalty completes the full Multiclass Support Vector Machine loss, which is made up of two components: the data loss (which is the average loss LiLi over all examples) and the regularization loss.
* hinge loss
* squared hinge loss SVM (or L2-SVM), which uses the form max(0,−)^2^
* Regularization
* Regularization penalty R(W)
* CNN
	- Convolutional Neural Networks are a powerful artificial neural network technique. They expect and preserve the spatial relationship between pixels in images by learning internal feature representations using small squares of input data. Feature are learned and used across the whole image, allowing for the objects in your images to be shifted or translated in the scene and still detectable by the network. There are ++three types of layers++ in a Convolutional Neural Network:
		- **Convolutional Layers** comprised of filters and feature maps.
		- **Pooling Layers** that down sample the activations from feature maps.
		- **Fully-Connected Layers** that plug on the end of the model and can be used to make predictions.
* Principal Component Analysis
	- It’s a method to reduce the dimensionality of a dataset. There are others, like Fisher transform
	-  It’s simpler and easier to understand or even visualize.
* Objective of Training
* Model
* Inference or prediction
* batch-size
* itration
* epoch
* ground-truth
* learning rate
* backward propogation
* Layers - convolution, Pooling (MaxPooling), fully-connected layer, flattening layer
* strides
* activation map
* kernet
* sigmoid neuron, RELU, TanH
* filter
* padding
* parameters
* placeholders, inputs
* optimization
* validation
* network design
* non-linear function activation functions
	* Sigmoid function
	>$ \sigma(x) = \frac{1}{1 + e^{-x}} $
* Overfitting
	- https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
	* Since we only have few examples, our number one concern should be overfitting. Overfitting happens when a model exposed to too few examples learns patterns that do not generalize to new data, i.e. when the model starts using irrelevant features for making predictions.
	* For instance, if you, as a human, only see three images of people who are lumberjacks, and three, images of people who are sailors, and among them only one lumberjack wears a cap, you might start thinking that wearing a cap is a sign of being a lumberjack as opposed to a sailor. You would then make a pretty lousy lumberjack/sailor classifier.
	* Data augmentation is one way to fight overfitting, but it isn't enough since our augmented samples are still highly correlated.
	*  Your main focus for fighting overfitting should be the **entropic capacity** of your model --how much information your model is allowed to store. A model that can store a lot of information has the potential to be more accurate by leveraging more features, but it is also more at risk to start storing irrelevant features. Meanwhile, a model that can only store a few features will have to focus on the most significant features found in the data, and these are more likely to be truly relevant and to generalize better.
* **Entropy**
	* https://en.wikipedia.org/wiki/Entropy_(information_theory)
	* Generally, information entropy is the average information of all possible outcomes.
	* There are different ways to modulate **entropic capacity**. The main one is the choice of **the number of parameters in your model**, i.e.
		- the number of layers
		- the size of each layer
		- weight regularization
			- such as L~1~ or L~2~ regularization, which consists in forcing model weights to taker smaller values.
		- Data augmentation
		- Dropout
			- Dropout also helps reduce overfitting, by preventing a layer from seeing twice the exact same pattern, thus acting in a way analoguous to data augmentation (you could say that both dropout and data augmentation tend to disrupt random correlations occuring in your data).
*  Dilated Convolutions

### Options in User Interface for Model Selection and Traning
**Deep Learning Key Terms**
1. Select Dataset
	* Dataset Summary:
		- Image Size: 256x256
		- Image Type: COLOR
		- DB backend: lmdb
		- Create DB (train): 4501 images
		- Create DB (val): 1499 images
2. Solver Options
	* Training epochs - How many passes through the training data?
	* Shuffle Train Data - For every epoch, shuffle the data before training
	* Snapshot interval (in epochs)
		- How many epochs of training between taking a snapshot?
	* Validation interval (in epochs)
		- How many epochs of training between running through one pass of the validation data?
	* Random seed
		- If you provide a random seed, then back-to-back runs with the same model and dataset should give identical results.
	* Batch size
		- How many images to process at once. If blank, values are used from the network definition. (accepts comma separated list)
	* Batch Accumulation
		- Accumulate gradients over multiple batches (useful when you need a bigger batch size for training but it doesn't fit in memory).
	* Solver type
		- What type of solver will be used?
		- NESTEROV: Nesterov's accelerated gradient (NAG)
		- ADAGRAD: Adaptive gradient (AdaGrad)
		- RMSPROP">RMSprop
		- ADADELTA">AdaDelta
		- ADAM">Adam
	* RMS decay value
		- If the gradient updates results in oscillations the gradient is reduced by times 1-rms_decay. Otherwise it will be increased by rms_decay.
	* Base Learning Rate
		- Affects how quickly the network learns. If you are getting NaN for your loss, you probably need to lower this value. (accepts comma separated list)
3. Data Transformations
	* Subtract Mean
		- Subtract the mean file or mean pixel for this dataset from each image.
		- None, Image, Pixel
	* Crop Size
		- If specified, during training a random square crop will be taken from the input image before using as input for the network.
	* Data Augmentations
		- Flipping: Randomly flips each image during batch preprocessing.
			* None, Horizontal, Vertical, Horizontal and/or Vertical
		- Quadrilateral Rotation - Randomly rotates (90 degree steps) each image during batch preprocessing.
			* None; 0, 90 or 270 degrees; 0 or 180 degrees; 0, 90, 180 or 270 degrees
		- Rotation (+- deg) (ARBITRARY_ROTATION) - "The uniform-random rotation angle that will be performed during batch preprocessing.
		- SCALING: Rescale (stddev) - Retaining image size, the image is rescaled with a +-stddev of this parameter. Suggested value is 0.07.
		- NOISE: Noise (stddev) - Adds AWGN (Additive White Gaussian Noise) during batch preprocessing, assuming [0 1] pixel-value range. Suggested value is 0.03.
		- HSV Shifting - Augmentation by normal-distributed random shifts in HSV color space, assuming [0 1] pixel-value range.
			* Hue: 0.02, Saturation: 0.04, Value: 0.06
4. Networks
	* Standard Networks
	* Previous Networks
	* Pretrained Networks
	* Custom Network
5. Framework
	* Caffe
	* Torch

### What are Convolutional Neural Networks and why are they important?
* https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/

In CNN terminology, the 3×3 matrix is called a ‘filter‘ or ‘kernel’ or ‘feature detector’ and the matrix formed by sliding the filter over the image and computing the dot product is called the ‘Convolved Feature’ or ‘Activation Map’ or the ‘Feature Map‘. It is important to note that filters acts as feature detectors from the original input image.

we can perform operations such as Edge Detection, Sharpen and Blur just by changing the numeric values of our filter matrix before the convolution operation [8] – this means that different filters can detect different features from an image, for example edges, curves etc. More such examples are available in Section 8.2.4 here.

In practice, a CNN learns the values of these filters on its own during the training process (although we still need to specify parameters such as number of filters, filter size, architecture of the network etc. before the training process). The more number of filters we have, the more image features get extracted and the better our network becomes at recognizing patterns in unseen images.

The size of the Feature Map (Convolved Feature) is controlled by three parameters that we need to decide before the convolution step is performed:
* Depth - Depth corresponds to the number of filters we use for the convolution operation.
* Stride - Stride is the number of pixels by which we slide our filter matrix over the input matrix.
* Zero-padding - Sometimes, it is convenient to pad the input matrix with zeros around the border, so that we can apply the filter to bordering elements of our input image matrix.

### Introducing Non Linearity (ReLU)
- **ReLU** stands for Rectified Linear Unit and is a non-linear operation. Other non linear functions such as tanh or sigmoid.
- ReLU is an element wise operation (applied per pixel) and replaces all negative pixel values in the feature map by zero.
- The purpose of ReLU is to introduce non-linearity in our ConvNet, since most of the real-world data we would want our ConvNet to learn would be non-linear
- Convolution is a linear operation – element wise matrix multiplication and addition, so we account for non-linearity by introducing a non-linear function like ReLU
- The ReLU operation applied to the feature maps provides the output feature map referred to as the **Rectified feature map**.

### The Pooling Step
- The function of Pooling is to progressively reduce the spatial size of the input representation
- Spatial Pooling (also called subsampling or downsampling) reduces the dimensionality of each feature map but retains the most important information.
- Spatial Pooling can be of different types: Max, Average, Sum etc.
- In case of Max Pooling, we define a spatial neighborhood (for example, a 2×2 window) and take the largest element from the rectified feature map within that window. Instead of taking the largest element we could also take the average (Average Pooling) or sum of all elements in that window. In practice, Max Pooling has been shown to work better.
- pooling operation is applied separately to each feature map (notice that, due to this, we will get three output maps from three input rectified feature maps.
- makes the input representations (feature dimension) smaller and more manageable
- reduces the number of parameters and computations in the network, therefore, controlling overfitting
- makes the network invariant to small transformations, distortions and translations in the input image (a small distortion in input will not change the output of Pooling – since we take the maximum / average value in a local neighborhood).
- helps us arrive at an almost scale invariant representation of our image (the exact term is “equivariant”). This is very powerful since we can detect objects in an image no matter where they are located 

**basic building blocks of any CNN**
- Convolution, ReLU & Pooling layers
- The output of the last Pooling Layer acts as an input to the Fully Connected Layer, which we will discuss in the next section.
- The output from the convolutional and pooling layers represent high-level features of the input image.
- In general, the more convolution steps we have, the more complicated features our network will be able to learn to recognize.
- In a traditional feedforward neural network we connect each input neuron to each output neuron in the next layer. That’s also called a fully connected layer, or affine layer. In CNNs we don’t do that. Instead, we use convolutions over the input layer to compute the output. This results in local connections, where each region of the input is connected to a neuron in the output. Each layer applies different filters, typically hundreds or thousands, and combines their results. During the training phase, a CNN automatically learns the values of its filters based on the task you want to perform.


### Fully Connected Layer
- The Fully Connected layer is a traditional Multi Layer Perceptron that uses a softmax activation function in the output layer (other classifiers like SVM can also be used)
- The purpose of the Fully Connected layer is to use the high-level features of the input image provided as an output from the convolutional and pooling layer, for classifying the input image into various classes based on the training dataset.
- The sum of output probabilities from the Fully Connected Layer is 1. This is ensured by using the Softmax as the activation function in the output layer of the Fully Connected Layer.

The **Softmax function** takes a vector of arbitrary real-valued scores and squashes it to a vector of values between zero and one that sum to one.

### Putting it all together – Training using Backpropagation
- Convolution + Pooling layers act as Feature Extractors from the input image while Fully Connected layer acts as a classifier.

### Data Divide
Split your training set into training set and a validation set. Use validation set to tune all hyperparameters. At the end run a single time on the test set and report performance.

Cross-validation. In cases where the size of your training data (and therefore also the validation data) might be small, people sometimes use a more sophisticated technique for hyperparameter tuning called cross-validation. Working with our previous example, the idea is that instead of arbitrarily picking the first 1000 datapoints to be the validation set and rest training set, you can get a better and less noisy estimate of how well a certain value of k works by iterating over different validation sets and averaging the performance across these. For example, in 5-fold cross-validation, we would split the training data into 5 equal folds, use 4 of them for training, and 1 for validation. We would then iterate over which fold is the validation fold, evaluate the performance, and finally average the performance across the different folds.


In practice. In practice, people prefer to avoid cross-validation in favor of having a single validation split, since cross-validation can be computationally expensive. The splits people tend to use is between 50%-90% of the training data for training and rest for validation.

It is worth considering some advantages and drawbacks of the Nearest Neighbor classifier. Clearly, one advantage is that it is very simple to implement and understand. Additionally, the classifier takes no time to train, since all that is required is to store and possibly index the training data. However, we pay that computational cost at test time, since classifying a test example requires a comparison to every single training example. This is backwards, since in practice we often care about the test time efficiency much more than the efficiency at training time. In fact, the deep neural networks we will develop later in this class shift this tradeoff to the other extreme: They are very expensive to train, but once the training is finished it is very cheap to classify a new test example. This mode of operation is much more desirable in practice.


## WebGL GPU based Deep Learning in Browser
* https://github.com/transcranial/keras-js
* https://news.ycombinator.com/item?id=12302932
* https://erkaman.github.io/regl-cnn/src/demo.html
* https://github.com/uber/horovod
* http://timdettmers.com/2017/04/09/which-gpu-for-deep-learning/

** Hardware Guides for Deep Learning
* http://timdettmers.com/2015/03/09/deep-learning-hardware-guide/

**References**
* http://www.deeplearningbook.org/
* https://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html
* https://in.udacity.com/course/how-to-use-git-and-github--ud775

## Pre-training, Transfer Learning
* http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html#transfer
* https://www.analyticsvidhya.com/blog/2017/06/transfer-learning-the-art-of-fine-tuning-a-pre-trained-model/
* https://www.analyticsvidhya.com/blog/2017/05/neural-network-from-scratch-in-python-and-r/
* https://github.com/alexgkendall/caffe-segnet/issues/3

## Model Architectures
* **VGG**
	- https://arxiv.org/abs/1409.1556
* **AlexNet**
* **LeNet**
	- one of the canonical network architectures for image classification
	-  how to implement LeNet in TensorFlow, highlighting data preparation, training and testing, and configuring convolutional, pooling, and fully-connected layers.
* **SqueezeNet**
	- https://arxiv.org/abs/1602.07360
	- https://github.com/DeepScale/SqueezeNet
	- a small CNN architecture called “SqueezeNet” that achieves AlexNet-level accuracy on ImageNet with 50x fewer parameters
	- https://gab41.lab41.org/lab41-reading-group-squeezenet-9b9d1d754c75
	- making a network smaller by starting with a smarter design versus using a clever compression scheme
	- Strategy 1. Make the network smaller by replacing 3x3 filters with 1x1 filters
		- https://iamaaditya.github.io/2016/03/one-by-one-convolution/
	- Strategy 2. Reduce the number of inputs for the remaining 3x3 filters
		- “squeeze” layers are convolution layers that are made up of only 1x1 filters
		-  “expand” layers are convolution layers with a mix of 1x1 and 3x3 filters.
		- By reducing the number of filters in the “squeeze” layer feeding into the “expand” layer, they are reducing the number of connections entering these 3x3 filters thus reducing the total number of parameters
		-  paper call this specific architecture the “fire module” and it serves as the basic building block for the SqueezeNet architecture.
	- Strategy 3. Downsample late in the network so that convolution layers have large activation maps.
		- The authors believe that by decreasing the stride with later convolution layers and thus creating a larger activation/feature map later in the network, classification accuracy actually increases
		- Having larger activation maps near the end of the network is in stark contrast to networks like VGG where activation maps get smaller as you get closer to the end of a network.
		- https://arxiv.org/abs/1412.1710
		- a delayed down sampling that leads to higher classification accuracy.
		- One of the surprising things I found with this architecture is the lack of fully-connected layers. What’s crazy about this is that typically in a network like VGG, the later fully connected layers learn the relationships between the earlier higher level features of a CNN and the classes the network is trying to identify. That is, the fully connected layers are the ones that learn that noses and ears make up a face, and wheels and lights indicate cars. However, in this architecture that extra learning step seems to be embedded within the transformations between various “fire modules”.
* **SqueezeNext: Hardware-Aware Neural Network Design**
	- https://arxiv.org/abs/1803.10615
* **NiN - Network In Network**
	- https://arxiv.org/abs/1312.4400
* **ResNet**
	- ResNet-18,ResNet-34
	- https://arxiv.org/abs/1512.03385
	- https://mc.ai/resnet-for-traffic-sign-classification-with-pytorch/\

### References
* http://cs.nyu.edu/~fergus/tutorials/deep_learning_cvpr12/
* https://ujjwalkarn.me/2016/08/09/quick-intro-neural-networks/
* http://mlss.tuebingen.mpg.de/2015/slides/fergus/Fergus_1.pdf
* https://github.com/rasbt/python-machine-learning-book/blob/master/faq/difference-deep-and-normal-learning.md
* https://www.quora.com/How-is-a-convolutional-neural-network-able-to-learn-invariant-features

#### Convolutional Neural Networks
* http://cs231n.github.io/convolutional-networks/
* http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/
* https://docs.gimp.org/en/plug-in-convmatrix.html
* http://colah.github.io/posts/2014-07-Understanding-Convolutions/

#### News/Articles
* http://www.dailymail.co.uk/sciencetech/article-3371075/See-world-eyes-driverless-car-town-Interactive-tool-reveals-autonomous-vehicles-navigate-streets.html
* http://www.sanborn.com/highly-automated-driving-maps-for-autonomous-vehicles/

## Machine Learning
* https://elitedatascience.com/learn-machine-learning
* https://www.datacamp.com/community/tutorials/deep-learning-python#gs.ny4aO4s

## Object Detection
* https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/blob/master/README.md
* https://www.pyimagesearch.com/2018/05/14/a-gentle-guide-to-deep-learning-object-detection/
* https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/
* https://www.pyimagesearch.com/2017/09/18/real-time-object-detection-with-deep-learning-and-opencv/
* https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/


## Street View Image Segmentation
**Segmentation**
* https://medium.com/nanonets/nanonets-how-to-use-deep-learning-when-you-have-limited-data-f68c0b512cab
* https://medium.com/nanonets/how-to-do-image-segmentation-using-deep-learning-c673cc5862ef
* https://blog.goodaudience.com/using-convolutional-neural-networks-for-image-segmentation-a-quick-intro-75bd68779225
* https://github.com/subodh-malgonde/semantic-segmentation
* http://www.cvlibs.net/datasets/kitti/eval_road.php
* https://github.com/udacity/CarND-Semantic-Segmentation/

### Feature_extraction_using_convolution
* http://deeplearning.stanford.edu/wiki/index.php/Feature_extraction_using_convolution

### Image Segmentations & Classification
The process of classifying each part of an image in different categories is called “image segmentation”.
* http://qucit.com/implementation-of-segnet/
* https://github.com/kjw0612/awesome-deep-vision

the last layer for a sigmoid one. The role of a softmax layer is to force the model to take a decision in a classification problem. Say you want to classify a pixel in one of three classes. A neural network will typically produce a vector of 3 probabilities, some of which can be close to each other, like [0.45, 0.38, 0.17].
But what you really want is just to know to which class this pixel belongs! Taking the maximum probability will give you a [1, 0, 0] vector which is what you want, but the max function isn’t differentiable, so your model can’t learn if you use it. A soft max is a kind of differentiable max, it won’t exactly give you a [1, 0, 0] vector, but something really close


The role of a sigmoid function is to output a value between 0 and 1, we use it to obtain the probability that a given pixel is a building pixel, thus obtaining something similar to a heatmap of this probability for each image.

* http://nghiaho.com/?p=1765
* https://devblogs.nvidia.com/parallelforall/exploring-spacenet-dataset-using-digits/
* https://aws.amazon.com/public-datasets/spacenet/
* http://nicolovaligi.com/converting-deep-learning-model-caffe-keras.html

**companies Involved**
*  DigitalGlobe, CosmiQ Works, and NVIDIA; SpaceNet

**References**
* [MULTI-SCALE CONTEXT AGGREGATION BY DILATED CONVOLUTIONS](https://arxiv.org/pdf/1511.07122.pdf)

The dilated convolution operator has been referred to in the past as “convolution with a dilated filter”

### Road Segmentation
* http://mi.eng.cam.ac.uk/projects/segnet/tutorial.html

**SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation**
* https://www.youtube.com/watch?v=G15Dg2QoI_M

**How to Simulate a Self-Driving Car**
* https://www.youtube.com/watch?v=EaY5QiZwSP4
* https://medium.com/self-driving-cars/term-1-in-depth-on-udacitys-self-driving-car-curriculum-ffcf46af0c08
* https://github.com/udacity/self-driving-car-sim
* https://hackernoon.com/five-skills-self-driving-companies-need-8546d2aba7c1
* https://developers.google.com/edu/c++/getting-started
* https://github.com/CPFL/Autoware

#### caffe-segnet

**Errors while installation**
* hdf5 dir not found HDF5_DIR-NOTFOUND
https://github.com/NVIDIA/DIGITS/issues/156

#### keras-segnet
* https://github.com/imlab-uiip/keras-segnet
```
sudo pip install pandas
```

model.add(Convolution2D(112,3,3, border_mode='same',init='uniform',input_shape=(136,136,3),dim_ordering='tf',name='conv_1.1'))
model.add(Conv2D(112,(3,3), border_mode='same',kernel_initializer='uniform',input_shape=(136,136,3),dim_ordering='tf',name='conv_1.1'))

https://github.com/tensorflow/cleverhans/issues/109

Convolutional2D -> CONV2D with following parameters have changed name/format:-
 Conv2D(10, 3, 3) becomes Conv2D(10, (3, 3))
 kernel_size can be set to an integer instead of a tuple, e.g. Conv2D(10, 3) is equivalent to Conv2D(10, (3, 3))
subsample -> strides
border_mode -> padding
nb_filter -> filters

## LeNet – Convolutional Neural Network in Python

* http://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/
* http://www.pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/

convolutional, activation, and pooling layers, fully-connected layer, activation, another fully-connected, and finally a softmax classifier

In many ways, LeNet + MNIST is the “Hello, World” equivalent of Deep Learning for image classification.

https://www.pyimagesearch.com/pyimagesearch-gurus/

- Common splits include the standard 60,000/10,000, 75%/25%, and 66.6%/33.3%. I’ll be using 2/3 of the data for training and 1/3 of the data for testing later in the blog post.

* http://opencv.org/
* https://medium.com/@acrosson/installing-nvidia-cuda-cudnn-tensorflow-and-keras-69bbf33dce8a
* http://chrisstrelioff.ws/sandbox/2014/06/04/install_and_setup_python_and_packages_on_ubuntu_14_04.html

In reality, an (image) convolution is simply an element-wise multiplication of two matrices followed by a sum.
* http://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/
* http://www.pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/
* http://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/
**An image is just a multi-dimensional matrix:**
-  image has a width (# of columns) and a height (# of rows), just like a matrix.
-  images also have a depth to them — the number of channels in the image.
-  for a standard RGB image, we have a depth of 3 — one channel for each of the Red, Green, and Blue channels, respectively.
- we can think of an image as a big matrix and kernel or convolutional matrix as a tiny matrix that is used for blurring, sharpening, edge detection, and other image processing functions.
- this tiny kernel sits on top of the big image and slides from left-to-right and top-to-bottom, applying a mathematical operation (i.e., a convolution) at each (x, y)-coordinate of the original image.
- blurring (average smoothing, Gaussian smoothing, median smoothing, etc.)
- edge detection (Laplacian, Sobel, Scharr, Prewitt, etc.), 
- sharpening
- all of these operations are forms of hand-defined kernels that are specifically designed to perform a particular function. is there a way to automatically learn these types of filters? And even use these filters for image classification and object detection?
- Convolution is simply the sum of element-wise matrix multiplication between the kernel and neighborhood that the kernel covers of the input image.
* Open Data for Deep Learning
	- https://deeplearning4j.org/opendata
	- https://www.kaggle.com/c/dogs-vs-cats
* http://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/
	- Neural Networks are essentially mathematical models to solve an optimization problem.
	- In this example, you can see that the weights are the property of the connection, i.e. each connection has a different weight value while bias is the property of the neuron.
	- The networks which have many hidden layers tend to be more accurate and are called deep network and hence machine learning algorithms which uses these deep networks are called deep learning.
	- Typically, all the neurons in one layer, do similar kind of mathematical operations and that’s how that a layer gets its name(Except for input and output layers as they do little mathematical operations)
* Convolutional Layer
* Pooling Layer
	- Pooling layer is mostly used immediately after the convolutional layer to reduce the spatial size(only width and height, not depth). This reduces the number of parameters, hence computation is reduced. 
	- The most common form of pooling is Max pooling where we take a filter of size $F*F$ and apply the maximum operation over the $F*F$ sized part of the image.
* Fully Connected Layer
	- If each neuron in a layer receives input from all the neurons in the previous layer, then this layer is called fully connected layer.
	- The output of this layer is computed by matrix multiplication followed by bias offset.

$cost=0.5\sum_{i=0}^n(y_{actual} - y_{prediction})^2$
- `~/.keras/keras.json`
- The objective of our training is to learn the correct values of weights/biases for all the neurons in the network that work to do classification between dog and cat.
- The Initial value of these weights can be taken anything but it works better if you take normal distributions(with mean zero and small variance).
- There are other methods to initialize the network but normal distribution is more prevalent.

## Learning Keras by Implementing the VGG Network From Scratch
https://hackernoon.com/learning-keras-by-implementing-vgg16-from-scratch-d036733f2d5

## Case Studies

### [MULTI-SCALE CONTEXT AGGREGATION BY DILATED CONVOLUTIONS](https://arxiv.org/pdf/1511.07122.pdf)

The dilated convolution operator has been referred to in the past as “convolution with a dilated filter”

* https://github.com/BVLC/caffe/issues/782
* https://github.com/BVLC/caffe/issues/263
* https://stackoverflow.com/questions/14585598/installing-numba-for-python

```
#python caffe path
#custom
export CAFFE_ROOT=$HOME/Documents/ml/caffe-segnet
#export PATH=$PATH:$CAFFE_ROOT/build/tools
export PYTHONPATH=$CAFFE_ROOT/python
# Install numba
sudo pip install numba
#
python predict.py dilation10_cityscapes.caffemodel 1.jpg
usage: predict.py [-h] [-o OUTPUT_PATH] [--gpu GPU]
                  [{pascal_voc,camvid,kitti,cityscapes}] [input_path]
predict.py: error: argument dataset: invalid choice: 'dilation10_cityscapes.caffemodel' (choose from 'pascal_voc', 'camvid', 'kitti', 'cityscapes')
#correct command
python predict.py pascal_voc 1.jpg
```

* https://github.com/richzhang/colorization/issues/2


Using CPU
[libprotobuf ERROR google/protobuf/text_format.cc:274] Error parsing text-format caffe.NetParameter: 297:13: Message type "caffe.ConvolutionParameter" has no field named "dilation".
WARNING: Logging before InitGoogleLogging() is written to STDERR
F0628 20:12:54.126731 13129 upgrade_proto.cpp:928] Check failed: ReadProtoFromTextFile(param_file, param) Failed to parse NetParameter file: models/dilation10_cityscapes_deploy.prototxt
*** Check failure stack trace: ***
Aborted (core dumped)

 Network initialization done.
[libprotobuf WARNING google/protobuf/io/coded_stream.cc:537] Reading dangerously large protocol message.  If the message turns out to be larger than 2147483647 bytes, parsing will be halted for security reasons.  To increase the limit (or to disable these warnings), see CodedInputStream::SetTotalBytesLimit() in google/protobuf/io/coded_stream.h.
[libprotobuf WARNING google/protobuf/io/coded_stream.cc:78] The total number of bytes read was 537496438
F0629 00:41:04.755961 18550 syncedmem.hpp:33] Check failed: *ptr host allocation of size 4464377856 failed
*** Check failure stack trace: ***
Aborted (core dumped)


**protobuff**
* https://github.com/google/protobuf/blob/master/src/README.md
* https://github.com/google/protobuf

### Snippets
```
python convert.py \
  --caffemodel=~/Documents/ml/dilation/pretrained/dilation8_pascal_voc.caffemodel \
  --code-output-path=./pascal_voc_tf/dil8_net.py \
  --data-output-path=./pascal_voc_tf/ \
  ~/Documents/ml/dilation/models/dilation8_pascal_voc_deploy.prototxt


python convert.py def_path=~/Documents/ml/dilation/models/dilation8_pascal_voc_deploy.prototxt --caffemodel=~/Documents/ml/dilation/pretrained/dilation8_pascal_voc.caffemodel --code-output-path=./pascal_voc_tf/dil8_net.py --data-output-path=./pascal_voc_tf/
```


## Nvidia Deep Learning

### What network architectures most closely resemble ones you use now?
* Alexnet
* GoogLeNet/Inception (v1, v2, v3, v4, v5, other variations)
* Resnet (18/50/101/152, other variations)
* VGG (16, 19, other variations)
* BigLSTM, OpenNMT
* DeepSpeach
* Faster RCNN variations
* YOLO/SSD variations
* SqueezeNet
* GAN


### Which of the following best describes your application area (choose one)? *
* Image and Video applications
* Signal and Speech applications
* Text and Document applications
* Multimodal (combination of the above)

### Which of the following would you say is the main bottleneck for deployment? *
* Throughput
* Latency
* Ease of deployment
* Ease of updating model
* Maintainability

### Nvidia Education

https://www.nvidia.com/en-us/deep-learning-ai/education/


deep learning courses
 - you’ll learn how to train, optimize, and deploy neural networks
accelerated computing courses,
  - you’ll learn how to assess, parallelize, optimize, and deploy GPU-accelerated computing applications

INSTRUCTOR-LED WORKSHOPS
ONLINE COURSES
ONLINE MINI COURSES


* Explore the fundamentals of deep learning for Computer Vision.
  - Explore the fundamentals of deep learning by training neural networks and using results to improve performance and capabilities.
  - Learn how to start solving problems with deep learning.
  - Learn how to train a deep neural network to recognize handwritten digits.
  - Explore how deep learning works and how it will change the future of computing.
  - Learn how to detect objects using computer vision and deep learning by identifying a purpose-built network and using end-to-end labeled data.
  - Implement common deep learning workflows such as Image Classification and Object Detection.
  - Experiment with data, training parameters, network structure, and other strategies to increase performance and capability.
  - Deploy your networks to start solving real-world problems
* Explore Fundamentals of Deep Learning for Multiple Data Types
* Explore how convolutional and recurrent neural networks can be combined to generate effective descriptions of content within images and video clips. 
* Learn to deploy deep learning to applications that recognize and detect images in real time.
* Learn how to design, train, and deploy deep neural networks for autonomous vehicles.
* Learn how to train a generative adversarial network (GAN) to generate images, explore techniques to make video style transfer, and train a denoiser for rendered images.
* Learn how to combine computer vision and natural language processing to describe scenes using deep learning.
* Learn how to make predictions from structured data.
* Learn how to classify both image and image-like data using deep learning by converting radio frequency (RF) signals into images to detect a weak signal corrupted by noise.


1. Fundamentals of Deep Learning for Computer Vision
  - Implement common deep learning workflows such as Image Classification and Object Detection.
  - Experiment with data, training parameters, network structure, and other strategies to increase performance and capability.
  - Deploy your networks to start solving real-world problems
  - On completion of this course, you will be able to start solving your own problems with deep learning
  * Learning Objectives: What You'll Learn
    - Identify the ingredients required to start a Deep Learning project.
    - Train a deep neural network to correctly classify images it has never seen before.
    - Deploy deep neural networks into applications.
    - Identify techniques for improving the performance of deep learning applications.
    - Assess the types of problems that are candidates for deep learning.
    - Modify neural networks to change their behavior.
2. Learn how to train a network using TensorFlow and the MSCOCO dataset to generate captions from images and video by:
  - Implementing deep learning workflows like image segmentation and text generation
  - Comparing and contrasting data types, workflows, and frameworks
  - Combining computer vision and natural language processing
  * Upon completion, you’ll be able to solve deep learning problems that require multiple types of data inputs
3. Deep Learning for Digital Content Creation Using GANs and Autoencoders
Learn techniques for designing, training, and deploying neural networks for digital content creation.
  - Train a Generative Adversarial Network (GAN) to generate images
  - Explore the architectural innovations and training techniques used to make arbitrary video style transfer
  - Train your own denoiser for rendered images
  - Upon completion of this course, you’ll be able to start creating digital assets using deep learning approaches
4. Deep Learning for Finance Trading Strategy
Linear techniques like principal component analysis (PCA) are the workhorses of creating ‘eigenportfolios’ for use in statistical arbitrage strategies. Other techniques using time series financial data are also prevalent. But now, trading strategies can be advanced with the power of deep neural networks.
  - Prepare time series data and test network performance using training and test datasets
  - Structure and train a LSTM network to accept vector inputs and make predictions
  - Use the Autoencoder as anomaly detector to create an arbitrage strategy
  - Upon completion, you’ll be able to use time series financial data to make predictions and exploit arbitrage using neural networks.



* AUTONOMOUS VEHICLES
* GAME DEVELOPMENT AND DIGITAL CONTENT
  - Deep Learning for Digital Content Creation Using GANs and Autoencoders Explore the latest techniques for designing, training, and deploying neural networks for digital content creation.


## Labs
https://nvidia.qwiklab.com/focuses/40?parent=catalog

* Free datasets are available from places like Kaggle.com and UCI. 
  - https://www.kaggle.com/datasets
  - https://archive.ics.uci.edu/ml/datasets.html
* Crowdsourced datasets are built through creative approaches - e.g. Facebook asking users to "tag" friends in their photos to create labeled facial recognition datasets
* More complex datasets are generated manually by experts - e.g. asking radiologists to label specific parts of the heart.

**Training vs. programming**
- The fundamental difference between artificial intelligence (AI) and traditional programing is that AI learns while traditional algorithms are programmed. 
- Artificial intelligence takes a different approach. Instead of providing instructions, we provide examples.
- We could show our robot thousands of labeled images of bread and thousands of labeled images of other objects and ask our robot to learn the difference. Our robot could then build its own program to identify new groups of pixels (images) as bread.

The "deep" in deep learning refers to many layers of artificial neurons, each of which contribute to the network's performance.
Processing huge datasets through deep networks is made possible by parallel processing, a task tailor made for the GPU.


**how do we expose artificial neural networks to data?**
**how to load data into a deep neural network to create a trained model that is capable of solving problems with what it learned, not what a programmer told it to do.**


Since a computer "sees" images as collections of pixel values, it can't do anything with visual data unless it learns what those pixels represent.


What if we could easily convert handwritten digits to the digital numbers they represent?

We could help the post office sort piles of mail by post code. This is the problem that motivated Yann LeCun. He and his team put together the dataset and neural network that we'll use today and painstakingly pioneered much of what we know now about deep learning.
We could help teachers by automatically grading math homework. This the problem that motivated the team at answer.ky, who used Yann's work to easily solve a real world problem using a workflow like what we'll work through now.
We could solve countless other challenges. What will you build?


http://yann.lecun.com/
http://answer.ky/

We're going to train a deep neural network to recognize handwritten digits 0-9. This challenge is called "image classification," where our network will be able to decide which image belongs to which class, or group.


It's important to note that this workflow is common to most image classification tasks, and is a great entry point to learning how to solve problems with Deep Learning.


Inside the folder train_small there were 10 subfolders, one for each class (0, 1, 2, 3, ..., 9). All of the handwritten training images of '0's are in the '0' folder, '1's are in the '1' folder, etc.
- This data is labeled. Each image in the dataset is paired with a label that informs the computer what number the image represents, 0-9. We're basically providing a question with its answer, or, as our network will see it, a desired output with each input. These are the "examples" that our network will learn from.
- Each image is simply a digit on a plain background. Image classification is the task of identifying the predominant object in an image. For a first attempt, we're using images that only contain one object. We'll build skills to deal with messier data in subsequent labs.
- http://yann.lecun.com/exdb/mnist/
- This data comes from the MNIST dataset which was created by Yann LeCun. It's largely considered the "Hello World," or introduction, to deep learning.

Also like the brain, these "networks" only become capable of solving problems with experience, in this case, interacting with data. 

Throughout this lab, we'll refer to "networks" as untrained artificial neural networks and "models" as what networks become once they are trained (through exposure to data).

For image classification (and some other tasks), DIGITS comes pre-loaded with award-winning networks.

However, to start, weighing the merits of different networks would be like arguing about the performance of different cars before driving for the first time. 

Building a network from scratch would be like building your own car. Let's drive first. We'll get there.

Creating a new model in DIGITS is a lot like creating a new dataset.


- Classification
- Segmentation
- Object Detection
- Processing
- Other

**epoch**
We need to tell the network how long we want it to train. An epoch is one trip through the entire training dataset. Set the number of Training Epochs to 5 to give our network enough time to learn something, but not take all day. This is a great setting to experiment with.

We need to define which network will learn from our data. Since we stuck with default settings in creating our dataset, our database is full of 256x256 color images. Select the network AlexNet, if only because it expects 256x256 color images.

* LeNet  Original paper [1998] 28x28 (gray)
  - http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
* AlexNet  Original paper [2012] 256x256 
  - http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional
* GoogLeNet  Original paper [2014] 256x256
  - http://arxiv.org/abs/1409.4842

We'll dig into this graph as a tool for improvement, but the bottom line is that after 5 minutes of training, we have built a model that can map images of handwritten digits to the number they represent with an accuracy of about 87%!


Inference
Now that our neural network has learned something, inference is the process of making decisions based on what was learned. The power of our trained model is that it can now classify unlabeled images.

test our trained model. 
- you can test a single image or a list of images.

It worked! (Try again if it didn't). You took an untrained neural network, exposed it to thousands of labeled images, and it now has the ability to accurately predict the class of unlabeled images. Congratulations!

Note that that same workflow would work with almost any image classification task. You could train AlexNet to classify images of dogs from images of cats, images of you from images of me, etc. If you have extra time at the end of this lab, theres another dataset with 101 different classes of images where you can experiment.


* https://jorditorres.org/research-teaching/tensorflow/first-contact-with-tensorflow-book/first-contact-with-tensorflow/
* http://deeplearning.net/tutorial/
* http://www.deeplearningitalia.com/wp-content/uploads/2017/12/Dropbox_Chollet.pdf

## PyImageSearch
* https://www.pyimagesearch.com/static/cv_dl_resource_guide.pdf
* https://www.pyimagesearch.com/2014/10/13/deep-learning-amazon-ec2-gpu-python-nolearn/

## TBD Notes
* **Drones**
- https://medium.com/nanonets/how-we-flew-a-drone-to-monitor-construction-projects-in-africa-using-deep-learning-b792f5c9c471

**Pedistrian Detection in Survelliance**
- https://github.com/thatbrguy/Pedestrian-Detector
- https://medium.com/nanonets/how-to-automate-surveillance-easily-with-deep-learning-4eb4fa0cd68d
* The most popular variants are the Faster RCNN, YOLO and the SSD networks
* There is always a Speed vs Accuracy vs Size trade-off when choosing an Object Detection algorithm.
* a scalable surveillance system should be able to interpret low quality images.
* Most high performance models consume a lot of memory
* Pocessing Power:
  - The video streams from the cameras are processed frame by frame on a remote server or a cluster.
  - The obvious problem is latency; you need a fast Internet connection for limited delay. Moreover, if you are not using a commerical API, the server setup and maintenance costs can be high.
  * By attaching a small microcontroller, we can perform realtime inference on the camera itself. There is no transmission delay, and abnormalities can be reported faster than the previous method. 
  * Moreover, this is an excellent add on for bots that are mobile, so that they need not be constrained by range of WiFi/Bluetooth available. (such as microdrones).
  * The disadvantage is that, microcontrollers aren’t as powerful as GPUs, and hence you may be forced to use models with lower accuracy.
  - https://medium.freecodecamp.org/how-to-play-quidditch-using-the-tensorflow-object-detection-api-b0742b99065d

  **Data Augmentation**
  - images in your camera feed maybe of lower quality. So you must train your model to work in such conditions.
  - https://medium.com/nanonets/how-to-use-deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced
  - add some noise to degrade the image quality of the dataset. We could also experiment with blur and erosion effects.

  **Datasets**
  - Towncenter
    * http://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/project.html#datasets

pip install -r requirements.txt
sudo apt-get install protobuf-compiler

protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

python create_tf_record.py \
    --data_dir=`pwd` \
    --output_dir=`pwd`

python object_detection/train.py \
--logtostderr \
--pipeline_config_path=pipeline.config \
--train_dir=train

python object_detection/inference.py \
--input_dir={PATH} \
--output_dir={PATH} \
--label_map={PATH} \
--frozen_graph={PATH} \
--num_output_classes=1 \
--n_jobs=1 \
--delay=0


**Commercial APIS**
- https://nanonets.com/
- https://github.com/NanoNets/object-detection-sample-python.git

**Transfer Learning**
- https://medium.com/nanonets/nanonets-how-to-use-deep-learning-when-you-have-limited-data-f68c0b512cab

**Pre-trained Models**
- https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
Download one of these models, and extract the contents into your base directory. You will receive the model checkpoints, a frozen inference graph, and a pipeline.config file.

**SSD**
- https://towardsdatascience.com/understanding-ssd-multibox-real-time-object-detection-in-deep-learning-495ef744fab

python object_detection/inference.py \
--input_dir=test_images \
--output_dir=test_images_output \
--label_map=annotations/label_map.pbtxt \
--frozen_graph=output/frozen_inference_graph.pb \
--num_output_classes=1 \
--n_jobs=1 \
--delay=0


## Numpy Tutorials
https://www.w3resource.com/python-exercises/numpy/python-numpy-exercise-39.php