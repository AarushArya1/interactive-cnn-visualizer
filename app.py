# first step 5/28: building a basic UI that only displays a title, a file uploader and a dropdown for example images

import streamlit as st
import os

# STAGE 1: ONLY A BASIC SHELL OF THE ACTUAL UI. ONLY IMAGE SELECTION IS AVAILABLE SO FAR. 
# MODEL, GRAD-CAM, ALL FEATURES WILL BE LINKED LATER

# HOW TO RUN:
# python -m streamlit run app.py


st.set_page_config(
    page_title = "Interactive CNN Visualizer",
    page_icon = "🤖",
    layout = "wide" # to use the full browser width
)

st.title("Interactive CNN Visualizer")
st.markdown(
    "FULL VERSION COMING SOON! For now, use the terminal based main.py and follow the instructions in the file comments. Later, you will be able to upload an image or choose an example to visualize how "
    "a pretrained ResNet-50 model makes predictions using Grad-CAM heatmaps. You will also be able to select various perturbations to add to the image, simulating a full ML experiment platform. "
)
st.divider()

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
            label = "Choose your example image",
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
    st.subheader("Your selected image is displayed before. Stay in touch for what is to come next for the project!")
    st.image(selected_image, use_container_width = False, width = 400) #we want to use our own width for the image, not the entire container width


    
         


    



