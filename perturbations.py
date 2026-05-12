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
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    # returns: A new PIL image with the noise applied
    return Image.fromarray(noisy_image)



# The following method rotates the image counter-clockwise by a given angle.
# Mild rotations like 15 degrees test slight viewpoint invariance for the model, while
# severe rotations such as 90 or 180 degrees can cause completely different predictions
def add_rotation(image, angle = 30):
    # returns a new PIL Image rotated by the given angle counter-clockwise
    return image.rotate(angle, expand=False, fillcolor = (0, 0, 0))


# The following method, add_occlusion, blacks out a rectangular region of the image
# I believe occlusions can be the most interpretable perturbation. Since if the prediction of the model changes
# after occluding a region, insights can be developed on the importance of that specific region.
# Feel free to experiment with add_occlusion and different rectangular regions to blur, especially for images belonging to real world applicable datasets

def add_occlusion(image, x = 80, y = 80, width = 64, height = 64):
    # x represents the left edge of the rectangle to blur out (in pixels from the left edge of the image). this value can be set. if not set, the default value is 80 pixels right of the left edge (which I believe is best after some rigorous testing)
    # y represents the top edge of the rectangle to blur out (in pixels from the top edge of the image). this value can be set. if not set, the default value is 80 pixels down from the top edge (which I believe is best after some rigorous testing)
    # width is the width of the occlusion rectangle in pixels. default is 64 pixels
    # height is the height of the rectangle in pixels. default is 64 pixels
    # This method returns a new PIL image with the rectangle blacked out
    
    image_np = np.array(image.copy())
    image_np[y : y + height, x : x + width] = 0
    return Image.fromarray(image_np)


    

# NOTE: To test perturbations.py, go to main.py and follow the testing instructions with the added perturbations
