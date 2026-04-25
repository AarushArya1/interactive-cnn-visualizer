# interactive-cnn-visualizer

This project is currently IN PROGRESS

This is an interactive experiment platform for visualizing and stress-testing CNN predictions with Grad-CAM (Gradient-weighted Class Activation Mapping). Explore trends within a CNN's decision making process. 

The EASIEST way to test and visualize HOW machine learning models make their decisions, and WHY models may be mispredicting, especially when data is noisy

Heatmaps essentially demonstrate which regions of the image are being used the most to make decisions (i.e. "warmer" regions.)

# How to use 

1. Upload your own image

- Upload image using the image uploader and the correct label. Avoid very low resolution (<224×224) images or images with abstract/unidentifiable subjects. Also avoid noisy images with heavy filters (that's our job)
- See the resulting detailed heatmap, as well as the model's prediction 
- If incorrect (correct label != predicted label), see a generated potential explanation 

2. Add different noise factors, simulating real-world conditions 

Choose from the settings menu: occlusions, blurs, rotations, etc
All images you now upload will be automatically transformed based on your selection(s)
See resulting side-by-side heatmaps: left is for the image without any noise, right is with the indicated noise

# Don't want to upload your own image? Here are some other options from the menu: 

- See stored examples and heatmaps for false predictions, and explanations of why
- See stored examples of heatmaps for false predictions with different perturbations (occlusions, blur, rotations, etc) alongside the unedited image, and analysis of how such noise impacts the model
- See stored examples of successful predictions, unique heatmaps, etc

# EXAMPLES OF SIGNIFICANT TRIALS & RESULTS:

- <i>THIS SECTION IS TO BE COMPLETED!<i>


# Want to learn more? 

- See background.md and (for research references) references.md 
- Model architecture: ResNet-50. See references.md 
- Future plans to expand this platform: ideas can include different model architectures, different accuracy levels, etc... Contact me at aarusharya@berkeley.edu






