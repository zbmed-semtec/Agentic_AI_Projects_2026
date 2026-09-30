# YOLOs11-ocr-license-plates

- Model ID: ahmedashrafhelmi/yolos
- License: MIT
- Tasks: computer vision, image, video, transfer learning, image text recognition, yolo, object detection
- Frameworks: Other
- Shared by: Ahmed Ashraf Helmi
- Source: https://www.kaggle.com/models/ahmedashrafhelmi/yolos

## Description

# Model Summary

YOLOv11m and YOLOv9m models for car plates detection and OCR recognition

## Usage

you can load models like the following:
`from ultralytics import YOLO`
`yolo_model = YOLO("model.pt")`

## System

yolo models for car plates detection, detect plates of cars at first step. Then, yolo model for OCR starts to recognize the arabic letters and numbers on the plates

They can be a part of smart parking systems or modern radars used in streets

## Implementation requirements

it takes about from 45 minutes to 1.5 hours for fine-tuning or training the models on custom dataset

# Model Characteristics

## Model initialization

fine-tuned from a pre-trained YOLOv11m and YOLOv9m models

## Model stats

size of the model is about 40 MBs with about 20million parameters and 600 layers

# Data Overview

a dataset for cars plates with labels of .txt files contains the bounding boxes and the second dataset of arabic letters and numbers 

## Training data

one of the datasets found on Kaggle and the second from Roboflow website which is the arabic numbers and letters

## References

model is finetuned on Arabic Car License Images on roboflow website
