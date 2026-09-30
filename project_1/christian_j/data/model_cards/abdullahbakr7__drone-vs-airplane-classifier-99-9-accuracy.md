# Drone vs Airplane Classifier 99.9% Accuracy

- Model ID: abdullahbakr7/drone-vs-airplane-classifier-99-9-accuracy
- License: MIT
- Tasks: computer vision, transfer learning, keras, resnet50-v2, object detection
- Frameworks: Keras
- Shared by: Abdullah Bakr
- Source: https://www.kaggle.com/models/abdullahbakr7/drone-vs-airplane-classifier-99-9-accuracy

## Description

# Model Summary

ResNet50-based binary classifier that distinguishes **drones from airplanes**. 
Built as a second-opinion layer for the YOLO26m drone detection pipeline 
triggered whenever YOLO detects a `drone` or `airplane`, overriding the label and confidence score.

## Usage

```python
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input
import cv2, numpy as np

classifier = load_model('drone_classifier_model.keras')

def predict(crop_bgr):
    image = cv2.resize(crop_bgr, (224, 224))
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    y_pred = classifier.predict(image, verbose=0)
    label = 'Drone' if np.argmax(y_pred) else 'Airplane'
    confidence = np.max(y_pred)
    return label, confidence
```

- **Input:** BGR image crop (H, W, 3) direct OpenCV output, any size (resized internally to 224×224)
- **Output:** tuple of (label string, confidence float in [0, 1])

## System

Part of a two-stage drone detection pipeline:
1. YOLO26m detects objects in the scene
2. This classifier re-evaluates any `drone` or `airplane` detection and overrides the result

Standalone use is also valid for any drone/airplane binary classification task.

## Implementation Requirements

- **Training hardware:** NVIDIA GeForce RTX 2060 SUPER (8GB)
- **Training time:** ~2 minutes (20 epochs, early stopping at best val_loss)
- **Inference:** CPU-compatible, ~10ms per crop on CPU

---

# Model Characteristics

## Model Initialization

Fine-tuned from ImageNet pretrained weights. The ResNet50 backbone is fully **frozen** only the final Dense(2, softmax) layer is trained (4,098 trainable parameters out of 23.5M total).

## Model Stats

| Property | Value |
|----------|-------|
| Total parameters | 23,591,810 |
| Trainable parameters | 4,098 |
| Model size | ~90 MB |
| Input shape | (1, 224, 224, 3) |
| Output shape | (1, 2) |

## Other Details

No pruning or quantization applied.

---

# Data Overview

## Training Data

- **Drones:** image crops extracted from the [YOLO Drone Detection Dataset](https://www.kaggle.com/datasets/muki2003/yolo-drone-detection-dataset) using YOLO bounding box annotations
- **Airplanes:** images from the Natural Images Dataset (airplane class)
- Preprocessing: resized to 224×224, ResNet50 `preprocess_input` normalization
- Augmentation: rotation (±45°), horizontal flip, brightness shift [0.8, 1.3]

## Demographic Groups

Not applicable object detection task with no human subjects.

## Evaluation Data

| Split | Size |
|-------|------|
| Train | ~72% |
| Validation | ~13% |
| Test | ~15% |

Stratified split to maintain class balance across all three sets.

---

# Evaluation Results

## Summary

| Metric | Score |
|--------|-------|
| Test Accuracy | **99.9%** |
| Loss function | sparse_categorical_crossentropy |

## Subgroup Evaluation

No formal subgroup analysis. Known limitation: performance may degrade on drone crops from very low resolution or extreme angles not represented in the training data.

## Fairness

Not applicable for this task.

## Usage Limitations

- Designed specifically for drone vs. airplane discrimination
- Performance depends on crop quality from the upstream YOLO detector
- Not tested on military drones or unconventional UAV shapes

## Ethics

Model is intended for airspace monitoring and safety applications. No sensitive demographic data involved. Misclassification risk exists at low confidence scores a confidence threshold should be applied in production.

## References

Model trained from scratch using ImageNet pretrained ResNet50 weights (via Keras Applications). Training data sourced from the YOLO Drone Detection Dataset (Kaggle) and the Natural Images Dataset (Kaggle)
