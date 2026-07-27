# Usage Instructions 

---

## Overview

This application is a visual exploration of image classification model interpretability and robustness. It allows users to explore how image classification models of different architectures make predictions, and more importantly, how their decision-making behavior changes under perturbations such as Gaussian noise, rotation, and occlusion.

## Getting Started

1. Choose a model architecture for your trial (ResNet-50, VGG-16, or EfficientNetB0). See background.
2. Choose an image. Either upload your own image or select an example image.
    - The best images for visualization have good lighting, high resolution, and have a clearly visible object with minimal background clutter.
    - However, it is recommended to test a wide range of images.
    - If you would like to select an example image instead, know that the catalog is extensive and contains a mix of basic images and more interesting images that may lead to interesting attention patterns.
3. Configure your trial by choosing perturbations.
    - You can also modify the settings of perturbations (e.g. Gaussian noise strength, rotation angle, occlusion box dimensions & location)
4. Click "Run" to generate the results.
    

## Interpreting the Results

1. The Gradient-weighted Class Activation Mapping (Grad-CAM) heatmap highlights regions that most influenced the model's prediction (these regions are warmer).
2. You can see the resulting Grad-CAM heatmaps for both your original image side-by-side with (if applicable) your image with your selected perturbations applied. This can allow you to visualize how the attention of your selected model changes due to the perturbations, from which you can make inferences as to how and why the predictions may have changed. 
3. The original predictions of the model (for the base image) and the predictions of the model after the image was perturbed are both shown. The confidence drop column shows the change in the model's confidence for each prediction after the image was perturbed. These can indicate a model's sensitivity to perturbations and issues in model robustness. 

## Try:

1. Different model architectures may focus on different image regions
2. Certain types of image perturbations may be the most severe for different architectures and/or different/more unique images
3. The most insightful heatmap comparisons may correspond to false predictions and/or huge confidence drops





 