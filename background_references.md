# Background, References, Tools Used

---

Want to learn how to use the application? See Instructions instead by clicking "How to Use".

This page simply provides a brief overview of the concepts in this project and how they were used. Please see References at the bottom of this page to learn more.

## Core Concepts & Grad-CAM
Convolutional Neural Networks (CNNs): CNNs are the baseline machine learning algorithms used to classify images. They consist of convolutional layers, where kernels (filters) highlights specific features in the image such as edges, curves, or colors and create feature maps. Activation functions are used to help the network focus only on notable positive features, and afterwards feature maps are downsampled to help with processing. After multiple stacks of these layers, a final classification layer assigns probabilities to what the image is.   

Gradient-weighted Class Activation Mapping (Grad-CAM): An Explainable AI technique which is the core of this project. Grad-CAM calculates the gradients of a specific prediction with respect to the features maps in the final convolutional layer. These gradients determine the importance of each feature map for the model's final prediction. They are then used to generate a visual heatmap highlighting the regions and features of the original image that the neural network most heavily relied on when making a decision. 

Warmer regions (red) in the visual heatmap (overlaid on the image) indicate more influential regions, cooler regions (blue) had little influence.

## Model Architectures & Training

This project uses three distinct CNN model architectures:

- ResNet-50: Uses skip connections to allow both the original input data and gradients to travel to later parts of the CNN. This prevents worsening performance for deeper networks. Widely used for image classification and this project's baseline.

- VGG-16: A uniform, repetitive architecture with a sequential stack of fully connected convolutional layers. Chosen for this project because the simple structure often produces different attention patterns from newer networks.

- EfficientNet-B0: Efficiently scales the network depth, width, and input resolution together in a balanced way. This leads to high classification accuracy with just a fraction of the computational resources used.

In this project (and many others), all three architectures use the IMAGENET1K_V1 pre-trained weights by PyTorch's torchvision library for classification models. These are trained on the ImageNet-1K database with over 1.2 million images spanning 1,000 distinct classes. 

## Image Perturbations

Users have the ability to select and modify three image perturbations (or changes). These reflect the type of noise present in image in various different real-world applications.  

Gaussian Noise: Creates a signal-like noise within the image. This noise has a probability density function equal to a normal distribution. For each pixel, a random value is chosen from a normal distribution centered at 0 with standard deviation = the noise strength (chosen by the user). The random value is then added to the pixel channel.

Rotation: Rotates the image counter-clockwise by an angle chosen by the user.

Occlusion: Fully blacks out a rectangular region of the image which removes all visual information in that region.The size and location of the obscured region is chosen by the user. This forces the model to rely on the remaining visible regions, so occlusions may be some of the most insightful perturbations.

    
## REFERENCES

1. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. Proceedings of the IEEE International Conference on Computer Vision (ICCV), 618–626. https://doi.org/10.1109/ICCV.2017.74
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770–778. https://doi.org/10.1109/CVPR.2016.90
3. Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. International Conference on Learning Representations (ICLR). https://arxiv.org/abs/1409.1556
4. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. Proceedings of the 36th International Conference on Machine Learning (ICML). https://arxiv.org/abs/1905.11946
5. Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). ImageNet: A Large-Scale Hierarchical Image Database. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 248–255. https://doi.org/10.1109/CVPR.2009.5206848
6. Paszke, A., Gross, S., Massa, F., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. Advances in Neural Information Processing Systems (NeurIPS), 32. https://arxiv.org/abs/1912.01703

# Tools Used

- PyTorch — The main deep learning framework used to load pretrained models (using Torchvision), run forward and backward passes for the CNNs, compute gradients for Grad-CAM
- Torchvision — Used to load pretrained ResNet-50, VGG-16, and EfficientNet-B0 models and apply standard ImageNet preprocessing transforms
- Streamlit — Used to build the front-end of the project, the interactive web application (including image upload, model selection, perturbation controls, results display, all webpage text)
Used to build the interactive web application, including image upload, model selection, perturbation controls, and results display.
- Pillow (PIL) — Used for all image operations: opening, converting, resizing, manipulating
- NumPy — Used for array operations when computing heatmaps or working with images
- OpenCV (cv2) — Used to apply the colormap COLORMAP_JET and blend the heatmap with the original image
- Python datetime & os modules — Used for generating timestamped output filenames in order to manage file paths across the project.


