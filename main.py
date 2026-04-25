import argparse
import os
from datetime import datetime
from PIL import Image
 
from model_ResNet50 import load_model, load_labels, preprocess_image, predict
from gradcam import generate_gradcam, overlay_heatmap_on_image
 
# THIS IS A TESTING FILE I MADE TO RUN THE GRAD-CAM
# TAKES IN THE INPUT IMAGE PATH AND THE TOP K PREDICTIONS WANTED BY THE USER
# THEN, PRINTS THE TOP K PREDICTIONS FROM RESNET_50
# MORE IMPORTANTLY, GENERATES THE CORRESPONDING GRAD-CAM HEATMAP FOR THE IMAGE AND SAVES IT IN the "outputs" folder

# HOW TO RUN:
#
#   1. Open your terminal and navigate to your project folder
# 
#   2. Run the script with an image path:
#        python main.py --image examples/test.jpg
#           or python3 on macOS/linux
#           use your own image path, or add something (or try something existing) to the examples folder of this project!

# ADD ANY IMAGES YOU WANT INTO EXAMPLES
#
#   Optional: you can also specify how many predictions to show (default is 5):
#        python main.py --image examples/[file_name].jpg --top_k 3 

#   If you need to help, you can run python myscript.py -h
#
#   OUTPUT: the Grad-CAM heatmap will be saved to the outputs/ folder
#           with a name like: outputs/[file_name]_gradcam_result_2026-04-24_11:19:52.jpg




# ArgumentParser is likely the best way for the user to enter the path to the input image, definitely
# way better than using a basic input scanner, especially due to the help menu that guides the user
# to better test the basic functionality of the Grad-CAM.

def parse_args():
    parser = argparse.ArgumentParser(
        description="Basic version of CNN Visualizer. Upload path to the input image, see top 5 model predictions, then see the visual Grad-CAM heatmap output for that image."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image (e.g. examples/dog.jpg)"
    )
    # decided to also add an argument K for the top K amount of predictions the user wants to display. Default is 5 
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="How many predictions would you like to display? (Default is 5)"
    )
    return parser.parse_args()
 
 
def make_output_filename(image_path):
   

    # Instead of saving the image to a randomly named file in the outputs folder, it would be better
    # to save the image according to how the image_path was named
    # example: if the image was named dog.jpg, the output should be dog_gradcam.jpg
    base_name = os.path.basename(image_path)
    base_of_imagepath = os.path.splitext(base_name)[0] # if basename is dog.jpg, baseofimagepath will be "dog"


    # Was curious and wanted to make the outputs folder a more clear record for when tests were run.
    # Therefore, I decided to use Python's datetime module, which gives the current date and time of testing
    # trying to format the current date and time into the output filename as well

    time_of_testing = datetime.now()
    timestamp = time_of_testing.strftime("%Y-%m-%d_%H:%M:%S")

    output_filename = os.path.join("outputs", f"{base_of_imagepath}_gradcam_result_{timestamp}.jpg")
    # example output filename: outputs/dog_gradcam_result_2026-04-24_11:19:52.jpg


    return output_filename 
 
 
def print_predictions(predictions):
    print("\nTop Predictions by ResNet50 Model:")
    for rank, (class_name, confidence) in enumerate(predictions, start=1):
        print(f"  {rank}. {class_name:<30} {confidence:.2f}%")
 
def main():
    args = parse_args()
 
    # Validate image path
    if not os.path.exists(args.image):
        print(f"We couldn't find an image at '{args.image}', please try again with a correct image")
        return
 
    
    os.makedirs("outputs", exist_ok=True) # in case the output folder doesn't exist (not sure why it wouldn't)
 
    print(f"\nThe image you have chosen is:  {args.image}")
 
    # everything below this point is very similar to the testing code in model_ResNet50.py and gradcam.py
    model = load_model()
    labels = load_labels()
    image_tensor = preprocess_image(args.image)
    predictions = predict(model, image_tensor, labels, top_k=args.top_k)
    print_predictions(predictions)
 
    heatmap = generate_gradcam(model, image_tensor)
    original_image = Image.open(args.image).convert("RGB")
    overlaid = overlay_heatmap_on_image(original_image, heatmap)
 
    
    output_path = make_output_filename(args.image)
    Image.fromarray(overlaid).save(output_path)
    print(f"Congrats, your heatmap has successfully been saved to: {output_path} Check it out in the outputs folder, and stay tuned for what is to come with this project!\n")

 
 
if __name__ == "__main__":
    main()