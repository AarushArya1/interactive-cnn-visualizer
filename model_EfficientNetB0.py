import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import urllib.request
import json
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# NOTE: Before reading, see notes in model_VGG16.py to understand the point of different model architectures.

# The third model architecture added for the purpose of comparison with ResNet50 and VGG16 is the EfficientNetB0 architecture
# How does it differ from ResNet50?
# ResNet50 uses residual connections to enable deeper connections, but EfficientNet-B0 is designed around the idea of efficient scaling by scaling the network depth, width, and input resolution together in a balanced way. 
# # By doing this, the network will use significantly less parameters and computation while achieving similar accuracy.

# So what to expect output-wise? Its much harder to say than for VGG-16 vs. ResNet50 (where the Grad-CAM will likely be less localized for VGG-16). 
# It will be much more nuanced and can help visually display how the efficiency-based design choice of EfficientNetB0 can influence confidence or patterns in the resulting Grad-CAM heatmap

LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
LABELS_PATH = "cnn_vis_imagenet_labels.json" 

def load_labels(): # Again, this is the same as in the initial model_ResNet50.py file, so look there
    
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)
    return labels

def load_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1) # another torchvision provided model which is great
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
    

# NOTE: TEST EFFICIENTNETB0 MODEL PREDICTION, UNCOMMENT 

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



       
                          
