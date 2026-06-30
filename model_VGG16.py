import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import urllib.request
import json
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# June 25th: deciding to add a second model architecture beyond just ResNet50 --VGG-16 -- to this project!
# this change will allow users to perform trials on and compare the results (Grad-CAM heatmaps and model predictions with or without perturbations)
# of different model architecture. The goal of this is for users to see how differing model architectures can affect model interpretability and robustness.
# 

# NOTE: My notes on VGG-16 background and difference for this project 

# The model architecture I decided to add for a basic comparison to ResNet50 is the VGG-16 architecture.
# VGG-16 is different since it is a straight sequential stack of convolutional layers with no shortcuts: in other words, data must flow from one layer to the next with no shortcuts
# On the other hand, ResNet50 uses skip connections, shortcuts where data can bypass layers. So each layer only needs to learn the difference/residual from its input

# What could this mean for the project?
# since VGG-16's layers builds directly on the previous one with no shortcuts, its feature maps are BROADER/LESS LOCALIZED then ResNet50.
# So therefore, the Grad-CAM heatmap will likely be broader/less hooked onto a specific region is what I am predicting but it will be interesting to see when app.py is configured with the different architectures.

# Note: almost everything in this file is the same as in model_ResNet50.py, except the model loading line of course and some preprocessing!
# NOTE: Please read the background above to understand how the visualizations will differ for VGG-16 vs. ResNet-50

LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
LABELS_PATH = "cnn_vis_imagenet_labels.json" 

def load_labels(): # Again, this is the same as in the initial model_ResNet50.py file, so look there
    
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)
    return labels

def load_model():
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1) # another torchvision provided model which is great
    model.eval()
    return model

def get_transform_object():
    return transforms.Compose([
        transforms.Resize(256), # shortest edge 256 px
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # these numbers are the mean and std of the entire imagenet training set 
            std=[0.229, 0.224, 0.225]
        )
    ])

def preprocess_image(image_source):

    if isinstance(image_source, Image.Image):
        image = image_source.convert("RGB")
    else:
        image = Image.open(image_source).convert("RGB")

    
    

    transform = get_transform_object()
    tensor = transform(image)
    return tensor.unsqueeze(0) 

# NOTE: important. something I didn't initially realize when implementing the new architectures:
# VGG 16, ResNet50, and (later) EfficientNetB0 each have a different final convolutional layer
# that is passed into gradcam.py.

def get_gradcam_layer(model):
    return model.features[-1]


# same as in model_ResNet50.py
def predict(model, image_tensor, labels, top_k=5):
    
    with torch.no_grad(): #NOTE: RIGHT NOW WE ARE ONLY PREDICTING, SO WE DON'T NEED TO TRACK THE GRADIENTS
        
        logits = model(image_tensor)

        probabilities = torch.softmax(logits, dim = 1)
        
        top_probabilities, top_classindices = torch.topk(probabilities, k=top_k, dim=1)
       
        results = []
        for prob, idx in zip(top_probabilities[0], top_classindices[0]):
            class_name = labels[idx.item()]
           
            confidence = round(prob.item() * 100, 2)
           
            results.append((class_name, confidence)) 
        return results
    

# NOTE: TEST VGG16 MODEL PREDICTION, UNCOMMENT 

#below is a way to test the model prediction. by running the file directly
# I am using a test image test.jpg of a red truck that I have dragged into the example folder
# If anyone wants to use this testing feature, feel free to change the path


if __name__ == "__main__":
   
    
    model = load_model()
    labels = load_labels()
 
    test_image_path = "examples/test.jpg"
    if os.path.exists(test_image_path):
        
        test_image_tensor = preprocess_image(test_image_path)
        predictions = predict(model, test_image_tensor, labels)
        #note to remember: predict returns the formatted array of [predicted_class_name, predicted_probability_value]
 
        print("TOP 5 PREDICTIONS BELOW:")
        for rank, (class_name, confidence) in enumerate(predictions, start=1):
            print(f"  {rank}. {class_name:<30} {confidence:.2f}%")
    else:
        print(f"We couldn't find a test image at the path'{test_image_path}'.")



       
                          
