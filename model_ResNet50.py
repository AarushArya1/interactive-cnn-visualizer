import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import urllib.request
import json
import os

# First step: create the model :D 
# DATE: April 16th, 2026

# NOTE FOR THE FUTURE! As I work through everything, 
# I will be try my best to document my entire thought & learning process to the best of my ability, 
# especially since my goal is to learn as much as I can through this process
# so in the future, I can look over this and actually understand with my own words how this process works


# I will be using ImageNet class labels for this, 
# this following database of simple labels seems to be one of the best I could find online.

LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"

# This is where I will store the labels locally
LABELS_PATH = "cnn_vis_imagenet_labels.json" #

def load_labels(): #goal here is to load the labels from the file as a python object, so it can be used in the future
    
    
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)

        # Okay so I needed to do a lot of research on how to actually get the labels as an object. 
        # What I didn't know is I needed to first open the file safely for reading
        # json.load gives the list of the ~1000 labels
    return labels

# Now that the labels are loaded, time to load the model

def load_model():
    # why did I choose Resnet50?
    # I did some research into the different architectures and
    # Most CNNs force all information through all layers, but ResNet-50 allows the gradient to flow 
    # through the network without forcing each layer to learn the entire input image again (i.e. whats different)
    # Therefore, it is very efficient 
    # Also Torchvision ships it pretrained on ImageNet, which is perfect for this first project
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()
    return model



# It took a while for me to understand what the following function actually returns :)
# get transform returns a callable function object, that when passed an image, applies the 
# following transformations and returns a tensor that the model can use for training
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

# applying the get transform method to the image, this part 
# was pretty intuitive for me
def preprocess_image(image_source):
    image = Image.open(image_source).convert("RGB")
    # the .convert("RGB") was something I was initially missing, it is important
    # since it handles cases like greyscale images and forces the image to the 3 RGB channels

    transform = get_transform_object()
    tensor = transform(image)
    return tensor.unsqueeze(0) #models expect a batch dimension??

    # we have converted the image to a tensor of RGB values (3 channels)

def predict(model, image_tensor, labels, top_k=5):
    # above: top_k = 5 means we will return a list of the top 5 classes the model thinks the image is

    with torch.no_grad(): #NOTE: RIGHT NOW WE ARE ONLY PREDICTING, SO WE DON'T NEED TO TRACK THE GRADIENTS
        
        logits = model(image_tensor)
        # logits are the resulting scores for every single class (not a probability yet)
        probabilities = torch.softmax(logits, dim = 1)
        # softmax converts the scores into probabilities between 0 and 1 for each class
        # basically confidence levels
        # should probably look into how it does this later on
        top_probabilities, top_classindices = torch.topk(probabilities, k=top_k, dim=1)
        # above: getting "topk" (top k values)

        #storing results
        results = []
        for prob, idx in zip(top_probabilities[0], top_classindices[0]):
            class_name = labels[idx.item()]
            # above: getting the class name from the labels list
            confidence = round(prob.item() * 100, 2)
            # converts to a percentage
            # the 2 at the end keeps 2 decimal places
        
            results.append((class_name, confidence)) # I just spent 30 min trying to figure out what was wrong and then I realized that this line was initially outside of the loop
        return results
    

# NOTE: TEST MODEL PREDICTION, UNCOMMENT 

#below is a way to test the model prediction. by running the file directly
# I am using a test image test.jpg of a red truck that I have dragged into the example folder
# If anyone wants to use this testing feature, feel free to change the path


# if __name__ == "__main__":
   
#     print("Time to load the model")
#     model = load_model()
#     labels = load_labels()
 
#     test_image_path = "examples/test.jpg"
#     if os.path.exists(test_image_path):
        
#         test_image_tensor = preprocess_image(test_image_path)
#         predictions = predict(model, test_image_tensor, labels)
#         #note to remember: predict returns the formatted array of [predicted_class_name, predicted_probability_value]
 
#         print("TOP 5 PREDICTIONS BELOW:")
#         for rank, (class_name, confidence) in enumerate(predictions, start=1):
#             print(f"  {rank}. {class_name:<30} {confidence:.2f}%")
#     else:
#         print(f"We couldn't find a test image at the path'{test_image_path}'.")

#test with the fire truck image: pretty good!
# TOP 5 PREDICTIONS BELOW:
# 1. fire engine                    65.45%
# 2. tow truck                      25.46%
# 3. pickup truck                   2.17%
# 4. snowplow                       1.62%
# 5. garbage truck                  1.23%#


       
                          






# print("This is the file for training the model.")

# print("Architecture used: ResNet-50")
