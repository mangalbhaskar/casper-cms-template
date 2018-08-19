/*
Title: Deep Learning Frameworks, Toolkits, Libraries
Decription: Deep Learning Frameworks, Toolkits, Libraries
Author: Bhaskar Mangal
Date: 24th-Jul-2018
Last update: 24th-Jul-2018
Tags: Deep Learning Frameworks, Toolkits, Libraries
*/

## Deep Learning Frameworks, Toolkits and Libraries
* https://dzone.com/articles/progressive-tools10-best-frameworks-and-libraries
* https://cloud.google.com/automl/
* https://aws.amazon.com/sagemaker/

**What framework(s) are avaiable to train deep learning models?**
**top-9-favorite-python-deep-learning-libraries**
* https://www.pyimagesearch.com/2016/06/27/my-top-9-favorite-python-deep-learning-libraries/

### Theano
- Theano is a numerical computation library for Python. In Theano, computations are expressed using a NumPy-esque syntax and compiled to run efficiently on either CPU or GPU architectures.
- Theano is a Python library for fast numerical computation to aid in the development of deep learning models. At it’s heart Theano is a compiler for mathematical expressions in Python. It knows how to take your structures and turn them into very efficient code that uses NumPy and efficient native libraries to run as fast as possible on CPUs or GPUs.

### PyTorch
* https://pytorch.org/
- PyTorch is an open source machine learning library for Python, based on Torch, used for applications such as natural language processing.

## [Caffe](http://caffe.berkeleyvision.org)
* https://github.com/BVLC/caffe
* http://caffe.berkeleyvision.org/
- Caffe is a deep learning framework made with expression, speed, and modularity in mind. It is developed by Berkeley AI Research (BAIR)/The Berkeley Vision and Learning Center (BVLC) and community contributors.
- Caffe (with DIGITS interface)

### Chainer
* https://chainer.org/

### Deeplearning4j: Open-source, Distributed Deep Learning for the JVM
- https://deeplearning4j.org/
- Eclipse Deeplearning4j is a deep learning programming library written for Java and the Java virtual machine and a computing framework with wide support for deep learning algorithms.

### Paddle Paddle
- http://paddlepaddle.org/

### MXNet
- https://mxnet.apache.org/
- Apache MXNet is a modern open-source deep learning framework used to train, and deploy deep neural networks

### MatConvNet
- www.vlfeat.org/matconvnet/
- MatConvNet is a MATLAB toolbox implementing Convolutional Neural Networks (CNNs) for computer vision applications

### TensorFlow
* TensorFlow is a Python library for fast numerical computing created and released by Google. Like Theano, TensorFlow is intended to be used to develop deep learning models. With the backing of Google, perhaps used in some of it’s production systems and used by the Google DeepMind research group, it is a platform that we cannot ignore. Unlike Theano, TensorFlow does have more of a production focus with a capability to run on CPUs, GPUs and even very large clusters.

### [Keras](https://keras.io/)
* https://keras.io/
* https://github.com/fchollet/keras
- Keras is an open source neural network library written in Python. It is capable of running on top of TensorFlow, Microsoft Cognitive Toolkit, Theano, or MXNet.
- A difficulty of both Theano and TensorFlow is that it can take a lot of code to create even very simple neural network models. These libraries were designed primarily as a platform for research and development more than for the practical concerns of applied deep learning. The Keras library addresses these concerns by providing a wrapper for both Theano and TensorFlow. It provides a clean and simple API that allows you to define and evaluate deep learning models in just a few lines of code.
- It’s a minimalist, modular neural network library that can use either Theano or TensorFlow as a backend.
- Keras is a high-level neural networks API, written in Python and capable of running on top of either TensorFlow or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.
```bash
sudo apt-get install python-numpy python-scipy -y
sudo apt-get install python-yaml -y
sudo apt-get install libhdf5-serial-dev -y
sudo pip install keras==1.0.8
```

### [scikit-learn](http://scikit-learn.org/stable/)
The scikit-learn library is a general purpose machine learning framework in Python built on top of SciPy. Scikit-learn excels at tasks such as evaluating model performance and optimizing model hyperparameters in just a few lines of code. Keras provides a wrapper class that allows you to use your deep learning models with scikit-learn.

```bash
sudo -H pip install -U scikit-learn
```

**Image processing in Python**
* http://scikit-image.org/
* https://scikits.appspot.com/scikit-image
	- scikit-image is a collection of algorithms for image processing
```bash
sudo -H pip install scikit-image
```

### [Caffe2](https://github.com/caffe2/caffe2)
* https://github.com/caffe2/caffe2
- Caffe2 is a deep learning framework enabling simple and flexible deep learning. Built on the original Caffe, Caffe2 is designed with expression, speed, and modularity in mind, allowing for a more flexible way to organize computation.

### ck-caffe
* https://github.com/dividiti/ck-caffe
CK-Caffe is an open framework for collaborative and reproducible optimisation of convolutional neural networks. It's based on the Caffe framework from the Berkeley Vision and Learning Center (BVLC) and the Collective Knowledge framework for customizable cross-platform builds and experimental workflows with JSON API from the cTuning Foundation (see CK intro for more details: 1, 2 ). In essence, CK-Caffe is an open-source suite of convenient wrappers and workflows with unified JSON API for simple and customized building, evaluation and multi-objective optimisation of various Caffe implementations (CPU, CUDA, OpenCL) across diverse platforms from mobile devices and IoT to supercomputers.

### TensorRT 4.0
NVIDIA TensorRT is a high-performance deep learning inference optimizer and runtime for deep learning applications.

TensorRT works across all NVIDIA GPUs using the CUDA platform. The following files are for use for Linux servers and workstations running NVIDIA Quadro, GeForce, and Tesla GPUs. NVIDIA recommends Tesla V100, P100, P4, and P40 GPUs for production deployment.

* Support for Text Translation and Natural Language Processing use cases (OpenNMT, GNMT)
* Support for Speech use cases (Deep Speech 2)


### Digits
* https://developer.nvidia.com/digits
- The NVIDIA Deep Learning GPU Training System (DIGITS) puts the power of deep learning into the hands of engineers and data scientists. DIGITS can be used to rapidly train the highly accurate deep neural network (DNNs) for image classification, segmentation and object detection tasks.
- DIGITS simplifies common deep learning tasks such as managing data, designing and training neural networks on multi-GPU systems, monitoring performance in real time with advanced visualizations, and selecting the best performing model from the results browser for deployment. DIGITS is completely interactive so that data scientists can focus on designing and training networks rather than programming and debugging.

## Toolkits, Libraries
* [dlib](http://dlib.net/) — a toolkit for real-world machine learning, computer vision, and data analysis in C++ (with Python bindings included, when appropriate).


## Installations

### **Caffe Install**
* https://github.com/BVLC/caffe
* http://caffe.berkeleyvision.org/installation.html
* Everything including caffe itself is packaged in 17.04 and higher versions. To install pre-compiled Caffe package, just do it by
```bash
sudo apt install caffe-cpu
```
**Dependencies**
* CUDA for GPU mode
	- CUDA compute capability  >= 3.0 (recommended)
	- CUDA 8 is required on Ubuntu 16.04
* BLAS
	- Caffe requires BLAS as the backend of its matrix and vector computations
	- There are several implementations of this library:
		* ATLAS:free, open source, and so the default for Caffe
		* MKL: commercial and optimized for Intel CPUs, with free licenses
		* OpenBLAS: free and open source; this optimized and parallel BLAS could require more effort to install, although it might offer a speedup
* Boost >= 1.55
* protobuf, glog, gflags, hdf5

**Optional dependencies:**
* OpenCV >= 2.4 including 3.0
* IO libraries: lmdb, leveldb (note: leveldb requires snappy)
* cuDNN for GPU acceleration (v6)

**Pycaffe:**
- For Python Caffe: Python 2.7 or Python 3.3+, numpy (>= 1.7), boost-provided boost.python

**CPU-only Caffe:**
- for cold-brewed CPU-only Caffe uncomment the CPU_ONLY := 1 flag in Makefile.config to configure and build Caffe without CUDA. This is helpful for cloud or cluster deployment.
- cmake build
- Ubuntu: http://caffe.berkeleyvision.org/install_apt.html

### caffe2
* https://github.com/caffe2/caffe2/issues/274
* https://techcrunch.com/2017/04/18/facebook-open-sources-caffe2-its-flexible-deep-learning-framework-of-choice/
* https://developer.nvidia.com/caffe2

```
# Clone Caffe2's source code from our Github repository
git clone --recursive https://github.com/pytorch/pytorch.git && cd pytorch
git submodule update --init
# Create a directory to put Caffe2's build files in
mkdir build && cd build
# Configure Caffe2's build
# This looks for packages on your machine and figures out which functionality
# to include in the Caffe2 installation. The output of this command is very
# useful in debugging.
cmake ..
# Compile, link, and install Caffe2
sudo make install
```

### Tensorflow
* [Learn Tensorflow](tensorflow.md)
**References**
* http://cv-tricks.com/artificial-intelligence/deep-learning/deep-learning-frameworks/tensorflow-tutorial/
* http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html
* https://stackoverflow.com/questions/26575587/cant-install-scipy-through-pip
* https://github.com/duguyue100/cs231n-practice/blob/master/cs231nlib/utils.py
* https://github.com/cs231n/cs231n.github.io
* http://vision.stanford.edu/teaching/cs231n/
* http://karpathy.github.io/2015/03/30/breaking-convnets/
* http://ccsubs.com/video/yt:GUtlrDbHhJM/cs231n-winter-2016-lecture-5-neural-networks-part-2-jhuz800c650-mp4/subtitles?lang=en
* https://devhub.io/repos/xiaohu2015-cs231n-assignment
* http://planetmath.org/vectorpnorm
* http://cs231n.github.io/convolutional-networks/
**Installation**
* Compilation from source
	* http://www.python36.com/install-tensorflow141-gpu/
	* https://www.tensorflow.org/install/install_sources

## DeepLearning Framework Installations
- https://medium.com/@vivek.yadav/deep-learning-setup-for-ubuntu-16-04-tensorflow-1-2-keras-opencv3-python3-cuda8-and-cudnn5-1-324438dd46f0