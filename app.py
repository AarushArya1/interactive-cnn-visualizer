
import streamlit as st
import os
from model_ResNet50 import load_model, load_labels, preprocess_image, predict
from PIL import Image 

from gradcam import generate_gradcam, overlay_heatmap_on_image

from perturbations import add_gaussian_noise, add_rotation, add_occlusion



# HOW TO RUN:
# python -m streamlit run app.py


st.set_page_config(
    page_title = "Interactive CNN Visualizer",
    page_icon = "🤖",
    layout = "wide" # to use the full browser width
)

# 
@st.cache_resource
def get_model():
    return load_model()
 
@st.cache_resource
def get_labels():
    return load_labels()
 
model = get_model()
labels = get_labels()

st.title("Interactive CNN Visualizer")
st.markdown(
    " INSTRUCTIONS TO BE INSERTED HERE. "
)
st.divider()

# Image selection is the first step

st.subheader("Select Image: follow the steps below")

input_method = st.radio(
    label = "Choose an image in one of the following ways: ",
    options = ["Upload your own image", "Choose an example image"],
    horizontal = True
)

selected_image = None
if input_method == "Upload your own image":
    uploaded_file = st.file_uploader(
        label="Upload an image",
        type=["jpg", "jpeg", "png"],
        help="We current support JPG, JPEG, PNG image formats"
    )
    if uploaded_file is not None:
        selected_image = uploaded_file
        st.success(f"You have successfully uploaded the following file: {uploaded_file.name}")

# the other case: load a file from examples
else:
    examples_directory = "examples"
    supported_extensions = {".jpg", ".jpeg", ".png"}
    if os.path.exists(examples_directory):
        example_files = sorted([
        f for f in os.listdir(examples_directory)
        if os.path.splitext(f)[1].lower() in supported_extensions
    ])
    else:
        example_files = []

    if example_files: 
        chosen = st.selectbox( # dropdown menu
            label = "Choose Your Example Image",
            options = example_files
        )   
        selected_image = os.path.join(examples_directory, chosen) # the join here essentially creates the full path to the image such as "examples/dog.jpg"
        # this is important since this FULL PATH is what needs to be passed to preprocess_image() when the model is loaded
        st.success(f"You have successfully chosen the following file: {chosen}")
    else:
        st.warning("Example folder has no images???")


# for now: just display the selected image!
if selected_image is not None:
    st.divider()
    st.subheader("Now, configure the settings of your trial. Use the controls below.")

    # There are now two additional inputs, not just one: the Top_K num predictions input and all the inputs for the perturbations.
    # Therefore, these controls will sit side by side in two columns.
    # Perturbation column will be wider since more controls

    topk_column, perturbation_column = st.columns([1, 2])

    with topk_column:
        st.markdown("Prediction Setting")
        top_k = st.number_input( # this is actually not a text box but instead a slider/widget
            label = "Enter the amount of top predictions from the model to display. After pressing predict, scroll down to see the resultant heatmap(s) and predictions!",
            min_value = 1,
            max_value = 20,
            value = 5, #default value
            step = 1,
            help = "how many predictions should the model return? From 1 to 20"
    
        )
    
    with perturbation_column:
        st.markdown("Apply perturbations. Perturbations reflect how models encounter adversarial data (corrupted images) in the real world.")
        st.caption("Current options include Gaussian Noise (apply noise of a certain level to each pixel), Rotation (of the image by certain degrees), and Occlusions (a black rectangle of your configuration is added to the image, hiding a certain part of the image). ")
        st.caption("Check any combination. In backend, stacks in the follow order: noise, rotation, occlusion")

        use_noise = st.checkbox("Gaussian Noise")
        if use_noise:
            noise_strength = st.slider(
                "Strength of Gaussian Noise (higher values lead to greater distortion within each image pixel)",
                min_value = 5, max_value = 100, value = 25, step = 5,
                help = "Higher values lead to greater distortion within each image pixel."

            )
        else:
            noise_strength = 0
        
        use_rotation = st.checkbox("Rotation")




    st.image(selected_image, use_container_width = False, width = 400) #we want to use our own width for the image, not the entire container width

# Next step (for now) is model prediction

st.divider()


# top_k: like in main.py, top_k is the number of predictions to generate



predict_clicked = st.button("Predict", type = "primary")

# now display predictions and grad cam. took me a time to handle edge cases where an image was actually not selected by the user. 
# but while doing so, i learned a lot from the streamlit api. even adder a spinner. I'll probably work on polishing this UI at the very end of this project
if predict_clicked:
    with st.spinner("Model running...Grad-CAM generating..."): # for future: Perturbated Grad-CAM generating...
        image = Image.open(selected_image).convert("RGB")
        image_tensor = preprocess_image(image)
        predictions = predict(model, image_tensor, labels, top_k = int(top_k))
        heatmap = generate_gradcam(model, image_tensor) # calling grad cam methods
        overlaid = overlay_heatmap_on_image(image, heatmap)


    st.divider()


    # NOTE: Initially, I was simply displaying all images below the last (i.e. the grad-cam)
    # result was right below the display of the picture that the user chose. I am going to change this.
    # With the addition of the perturbations, the results will be displayed in a 4x4 Grid.
    # - With top left being chosen image
    # - top right being original image with all perturbations applied
    # - bottom left being Grad-CAM for image with no perturbations
    # - bottom right being Grad-CAM for image with all perturbations.


    # the generated Grad-CAM heatmap is going to split the screen with the original image.
    # therefore, I am using st.columns to split the page into side by side sections, where the original image will be on the right and the Grad-CAM on the left
    # the 0.05 parameter adds a spacer column between the split.

    col_original, col_spacer, col_heatmap = st.columns([1, 0.05, 1]) # original image, spacing, heatmap respectively

    with col_original:
            st.markdown("YOUR ORIGINAL IMAGE")
            st.image(image, use_container_width=True)
 
    with col_heatmap:
        st.markdown("GENERATED Grad-CAM HEATMAP")
        st.image(overlaid, use_container_width=True)
        st.header("What does the Grad-CAM mean?")
        st.caption(
            "Red regions are what most influenced the model's prediction."
            "Blue regions had the LEAST influence on the model's prediction."
            "To learn more about how Grad-CAM (or the rest of this project) works, see References or Background from the menu bar" # for future implementation btw 
        )

    # Predictions

    st.divider()
    st.markdown(f"TOP {top_k} PREDICTIONS:")
    for rank, (class_name, confidence) in enumerate(predictions, start = 1):
        st.markdown(f"**{rank}.** {class_name}")
        st.progress(confidence / 100) # this is a progress bar which is filled to the confidence percentage for that prediction. 
        st.caption(f"{confidence:.2f}%")

    

st.divider()
st.subheader("MORE FEATURES COMING SOON! Most notably, this will include an option to choose and analyze key example outputs instead of selecting your own image. A download feature will also be added. I also hope to add significantly more labels to feed into the model for a greater and more realistic variety for classification. This will be a polished app very soon, so STAY IN TOUCH with this project.")
feedback = st.text_input("Questions? Email me at aarusharya@berkeley.edu")







# note for the future

# grad cam heatmap should be displayed nicely in a box in the middle of the screen (need to figure out streamlit), ABOVE the display of the original selected image. a nice label to seperate these two.
# instead of select an image, there will be a completely other option (highest layer) to instead view example outputs

# need a download result feature

# add more to the examples menu, standardize this project and the file structure (at the end, this is a later step)
# and format the "or choose one of our images" and  "choose an example output" to look nice -- like those images are actually there. so its visually appealing. will need to find a way to do this later on.

# oh yeah also should have a menu bar with other options like references, background, about (basically readme.md) and need to populate

# SHOULD FEED IN MORE LABELS TO THE MODEL!!!! SO IT CAN DO OTHER, MORE COMPLEX STUFF





    



