# regnety

- Model ID: adityakane/regnety
- License: Apache 2.0
- Tasks: image, image classification, english
- Frameworks: TensorFlow2, TfLite
- Shared by: Aditya Kane
- Source: https://www.kaggle.com/models/adityakane/regnety

## Description

## Overview

This collection contains RegNetY classifiers and feature extractors trained on ImageNet-1k. They can be used for out-of-the box inference as well as fine-tuning. A detailed tutorial is available as a Colab Notebook at . Codebase used for training these models is available here.


## Table of contents

| Model Name    | Accuracy | Inference speed on K80 | Inference speed on V100 | FLOPs (Number of parameters) | Link                                                                                                                                                                 |
|---------------|----------|------------------------|-------------------------|------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| RegNetY 200MF | 67.54%   | 656.626 images/sec     | 591.734 images/sec      | 200MF (3.23 million)         | [Classifier](/models/adityakane/regnety/frameworks/TensorFlow2/variations/200mf-classification/versions/1), [Feature Extractor](/models/adityakane/regnety/frameworks/TensorFlow2/variations/200mf-feature-extractor/versions/1) |
| RegNetY 400MF | 70.19%   | 433.874 images/sec     | 703.797 images/sec      | 400MF (4.05 million)         | [Classifier](/models/adityakane/regnety/frameworks/TensorFlow2/variations/400mf-classification/versions/1), [Feature Extractor](/models/adityakane/regnety/frameworks/TensorFlow2/variations/400mf-feature-extractor/versions/1) |
| RegNetY 600MF | 73.18%   | 359.797 images/sec     | 921.560 images/sec      | 600MF (6.21 million)         | [Classifier](/models/adityakane/regnety/frameworks/TensorFlow2/variations/600mf-classification/versions/1), [Feature Extractor](/models/adityakane/regnety/frameworks/TensorFlow2/variations/600mf-feature-extractor/versions/1) |
| RegNetY 800MF | 73.94%   | 306.270 images/sec     | 907.439 images/sec      | 800MF (6.5 million)          | [Classifier](/models/adityakane/regnety/frameworks/TensorFlow2/variations/800mf-classification/versions/1), [Feature Extractor](/models/adityakane/regnety/frameworks/TensorFlow2/variations/800mf-feature-extractor/versions/1) |



MF signifies million floating point operations.

Reported accuracies are measured on ImageNet-1k validation dataset.

## References 

[1] [Designing Network Design Spaces by Radosavovic et al](https://arxiv.org/abs/2003.13678).  
[2] [ImageNet-1k](https://www.image-net.org/challenges/LSVRC/2012/index.php)  
[3] [Colab tutorial](https://colab.research.google.com/github/AdityaKane2001/regnety/blob/main/RegNetY_models_in_TF_2_5.ipynb)   
[4] [AdityaKane2001/regnety](https://github.com/AdityaKane2001/regnety)

## References

https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs/adityakane2001/collections/regnety/1.md
