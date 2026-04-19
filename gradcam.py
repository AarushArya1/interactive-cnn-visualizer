import torch
import torch.nn.functional as F
import numpy as np
import cv2


# Grad-CAM essentially works by slightly changing 
# the activations in the last convolutional layer of the CNN,
# and seeing how much the final prediction score would change. 
# The regions that cause the biggest change are the regions that the model was using the most to make the prediction

# so:
# 1. Run forward pass and grad activations in layer4 
# Since layer4 is the last layer in ResNet-50
# 2. Run a backward pass from the predicted class score, get the gradients flowing into layer4

def generate_gradcam(model, image_tensor, class_idx=None):
    #explanation of method:
    # image_tensor is the preprocessed tensor for the given image, of shape (1, 3, 224, 224)
    # class_idx is none because we will use the top predicted class automatically
    
    #returns a 2d numpy array (224, 224) with values [0, 1]
    # higher values --> regions that most influenced the prediction
    # Later, we will use the returned array to create the heatmap

    activations = []
    gradients = []

    def forward_hook(module, input, output):
        # Didn't realize that PyTorch enforces a fixed signature for hook functions
        activations.append(output)

    #Similarly, for the backward pass (to get the gradients)

    def backward_hook(module, grad_input, grad_output):
        #Again, fixed signature
        gradients.append(grad_output[0])
        #The above tells us how much each activation in layer 4 actually influenced the final prediction score

    forward_handle = model.layer4.register_forward_hook(forward_hook)
    backward_handle = model.layer4.register_full_backward_hook(backward_hook)

    model.eval()
    logits = model(image_tensor)
    # logits are the resulting scores for every single class (not a probability yet)
    if class_idx is None:
        class_idx = logits.argmax(dim=1).item()
        # class_idx is now the class with the highest resulting score
    
    model.zero_grad() #clears gradients left over previously
    target = torch.zeros_like(logits)
    target[0, class_idx] = 1.0
    # the gradient signal is 1 at the target class (class_idx) and 0 everywhere else

    logits.backward(gradient=target)

    forward_handle.remove()
    backward_handle.remove()
    #remove hooks afterwards to avoid wasting memory

    # COMPUTING THE ACTUAL HEATMAP

    activation = activations[0]   # Shape: (1, 512, 7, 7)
    gradient = gradients[0]       # Shape: (1, 512, 7, 7)

    weights = gradient.mean(dim=[2, 3], keepdim=True)   
    # averages gradients across the grid, giving one weight per channel
    weighted_sum = (weights * activation).sum(dim=1, keepdim=True)  # Shape: (1, 1, 7, 7)
    #below, the relu function zeroes out all negative values (we only care about regions that increased class score)
    heatmap_raw = F.relu(weighted_sum)

    # This is what I initially missed. I need to resize heatmap to be 224x224

    heatmap_resized = F.interpolate(heatmap_raw, size=(224, 224), mode="bilinear", align_corners=False)
    heatmap_np = heatmap_resized.squeeze().detach().numpy()

    heatmap_max = heatmap_np.max()
    heatmap_min = heatmap_np.min()
    #max and min values

    # Important error correction: need this condition, since if the heatmap is flat, heatmap_max = heatmap_min and therefore, will be dividing by 0
    
    heatmap_np = (heatmap_np - heatmap_min) / (heatmap_max - heatmap_min)
    

    return heatmap_np

def overlay_heatmap_on_image(original_image, heatmap, a = 0.45):
    # parameter a is the blend strength of the heatmap overlay. I spent some time experimenting with different values, 0.45 seemed to be ideal

    image_np = np.array(original_image.resize((224, 224)))
    # the original image now matches the heatmap

    # NOTE: red = high activation, blue = low activation

    heatmap_uint8 = np.uint8(255 * heatmap)                         
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  

    # Learned this a bit too late: CV2 uses a different color channel order than Streamlit
    # So for the actual visual, we need to convert to RGB

    
    final_overlaid_heatmap = cv2.addWeighted(image_np, 1 - a, colored_heatmap, a, 0)
    
    # how the above works:
    # a is how strong the heatmap shows through, so the original image has strength of 1-a while the heatmap has strength of a

    return final_overlaid_heatmap




# TESTING: UNCOMMENT CODE AND RUN THIS FILE DIRECTLY
# Use image in examples or own path to image
# Also requires a working model: see model_ResNet50.py
# This code will save the heatmap overlay to outputs/gradcam_test.jpg

if __name__ == "__main__":
    
    import os
    from PIL import Image
    from model_ResNet50 import load_model, load_labels, preprocess_image, predict
 
    test_image_path = "examples/test.jpg" #REPLACE
 
    if not os.path.exists(test_image_path):
        print("We couldn't find a test image at that path.")
    else:
        model = load_model()
        labels = load_labels()
        image_tensor = preprocess_image(test_image_path)
 
        predictions = predict(model, image_tensor, labels)
        top_class, top_confidence = predictions[0]
        print(f"TOP PREDICTION: {top_class} ({top_confidence:.2f}%)")

        # above is basically just prediction code all over again
        # generating heatmap BELOW
        heatmap = generate_gradcam(model, image_tensor)
 
        original_image = Image.open(test_image_path).convert("RGB")
        overlaid = overlay_heatmap_on_image(original_image, heatmap)
 
        os.makedirs("outputs", exist_ok=True)
        output_path = "outputs/gradcam_test.jpg"
        Image.fromarray(overlaid).save(output_path)
    
        # NOW THE IMAGE IS SUCCESSFULLY SAVED TO THE OUTPUT PATH












