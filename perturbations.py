import numpy as np
from PIL import Image, ImageFilter

# This file is to implement various perturbations on an image.
# The file takes in a PIL (python imaging library) image, applies perturbations, and returns the perturbed PIL image as output

# The good news is, this allows perturbations.py to work seemlessly with the rest of the pipeline
# because the preprocess_image() function in model_ResNet50.py accepts a PIL image
# Therefore, with this approach, it is very easy to call any of the below perturbation functions on an image before sending the perturbed image to the model/gradcam

# pipeline: PIL image → perturbation function → PIL image → preprocess_image() → image_tensor → generate_gradcam()

def add_gaussian_noise(image, strength = 25):
    # Adds random Gaussian noise to every pixel in the image
    # Background research I did on how this works:

    # For each pixel, a random value is chosen from a normal distribution centered at 0 with standard deviation = strength 
    # then the random value is added to the pixel channel, creating signal-like noise within each image. 
    # in real life, it arises from natural processes like thermal vibration

    # note the strength: larger strength (i.e. 80) is extremely heavy distortion
    # CHANGE THE STRENGTH PARAMETER FOR DIFFERENT RESULTS!

    image_np = np.array(image, dtype=np.float32)
    # float32 so we can add decimal noise values easily
    noise = np.random.normal(loc = 0, scale = strength, size = image_np.shape)
    noisy_image = image_np + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    # returns: A new PIL image with the noise applied
    return Image.fromarray(noisy)

# For the future
def add_another_pertubation(image, another_parameter):
    pass


# NOTE: To test perturbations.py, go to main.py and follow the testing instructions with the added perturbations


    




