# Handwritten Digit Recognition

This project implements a neural network model for recognizing handwritten digits using the MNIST dataset. It includes both single-threaded and multi-threaded versions of the implementation.

## Table of Contents
1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Implementations](#implementations)
6. [Performance](#performance)

## Overview

This project uses TensorFlow to create and train a simple neural network for classifying handwritten digits. The MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits (0-9), is used for training and evaluation.

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy

## Installation

1. Clone this repository:
git clone https://github.com/yourusername/handwritten-digit-recognition.git
cd handwritten-digit-recognition
2. Install the required packages:
 pip install tensorflow numpy
## Usage

To run the single-threaded version:
python single_threaded.py
To run the multi-threaded version:
python multi_threaded.py
## Implementations

### Single-Threaded Version

The single-threaded version (`single_threaded.py`) uses TensorFlow's standard APIs to load the MNIST dataset, preprocess the data, define and train the model, and evaluate its performance.

### Multi-Threaded Version

The multi-threaded version (`multi_threaded.py`) utilizes Python's threading capabilities to parallelize data preprocessing and augmentation. It creates a pool of worker threads to apply data augmentation techniques to the training data before model training begins.

## Performance

The performance of both implementations can be compared based on the training duration and the final test accuracy. The multi-threaded version may show improved performance, especially on systems with multiple CPU cores, though the exact improvement will depend on the specific hardware and the complexity of the data preprocessing steps.

Note: Due to the simplicity of the MNIST dataset and the data augmentation techniques used in this example, the performance difference between the single-threaded and multi-threaded versions may not be significant. More complex datasets and preprocessing steps would likely show a greater benefit from multi-threading.