# disease-classification

- Model ID: agripredict/disease-classification
- License: BSD-3-Clause
- Tasks: image, image classification, english
- Frameworks: TensorFlow2, TfLite
- Shared by: AgriPredict
- Source: https://www.kaggle.com/models/agripredict/disease-classification

## Description

## Overview
Agripredict sets to leverage the use of machine learning to diagnose common plant pests and diseases. [About AgriPredict](https://agripredict.com)

The model uses the MobileNet V1 architecture, covering diseases for crops such as Maize (Corn), Soy, Cabbage and Tomatoes. It is trained on an internal AgriPredict dataset containing over 14 classes. AgriPredict intends to publish more versions of the model in the future with better accuracy and performance.

## Usage
### Input

This model takes input of images.

* Inputs are expected to be 3-channel RGB color images of size 300 x 300, scaled to 1./255.
Shape: (300, 300, 3)

### Output

This model outputs to a vector of (None, 10, 10,), corresponding to
        &lt;['Tomato Healthy', 'Tomato Septoria Leaf Spot',
       'Tomato Bacterial Spot', 'Tomato Blight', 'Cabbage Healthy',
       'Tomato Spider Mite', 'Tomato Leaf Mold',
       'Tomato_Yellow Leaf Curl Virus', 'Soy_Frogeye_Leaf_Spot',
       'Soy_Downy_Mildew', 'Maize_Ravi_Corn_Rust', 'Maize_Healthy',
       'Maize_Grey_Leaf_Spot', 'Maize_Lethal_Necrosis', 'Soy_Healthy',
       'Cabbage Black Rot']&gt;.

#### References
[1] Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., … Adam, H. (2017). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. Retrieved from http://arxiv.org/abs/1704.04861

## References

https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs/agripredict
