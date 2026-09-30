# Deepfake Detection

- Model ID: aestroe/deepfake-detection
- License: MIT
- Tasks: india, china, north america, asia, beginner, intermediate, advanced, data analytics, deep learning, neural networks, transfer learning, japan, audio classification, inception v3, transformer, video classification
- Frameworks: TensorFlow2
- Shared by: HIMANSHU BANSAL
- Source: https://www.kaggle.com/models/aestroe/deepfake-detection

## Description

- Image Detection: Leverages transfer learning with advanced base models such as XceptionV3, InceptionV3, ResNet, and DenseNet to extract and identify complex image features. Ensemble methods are employed to combine predictions from these models, effectively reducing bias toward specific classes and improving accuracy.

- Video Detection: Processes video data by extracting one frame per second. Each extracted frame is fed into the image detection model, and a voting mechanism is applied to aggregate the results across frames, ensuring a reliable final prediction.

- Audio Detection: Converts audio data into mel spectrograms, enabling the model to analyze time-frequency features. These spectrograms are processed by a Convolutional Neural Network (CNN) to detect unique patterns indicative of manipulated or synthetic audio content.# Model Summary

## References

The model was trained using the following datasets:

Image Datasets
- 140K Real and Fake Faces Dataset
- Deepfake and Real Images Dataset

Audio Datasets
- ASVspoof 2019 Dataset
- In the Wild (Audio Deepfake)
