
import streamlit as st
import os
from model_ResNet50 import load_model, load_labels, preprocess_image, predict
from PIL import Image 

# 5/28: ONLY A BASIC SHELL OF THE ACTUAL UI. ONLY IMAGE SELECTION AND BASIC MODEL PREDICTION IS AVAILABLE SO FAR. 
# GRAD-CAM AND ALL FEATURES WILL BE LINKED LATER

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
    "FULL VERSION COMING SOON! You may use this UI to only upload images and generate predictions. To fully test this project currently, use the terminal based main.py and follow the instructions in the file comments. Later, you will be able to upload an image or choose an example to visualize how "
    "a pretrained ResNet-50 model makes predictions using Grad-CAM heatmaps. You will also be able to select various perturbations to add to the image, simulating a full ML experiment platform. "
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
    st.subheader("Your selected image is displayed before. Now, add perturbations (optional) and press predict. Or, select a different image (or use one of our examples)")
    st.image(selected_image, use_container_width = False, width = 400) #we want to use our own width for the image, not the entire container width

# Next step (for now) is model prediction

st.divider()
st.subheader("ResNet50 Model Prediction")

# top_k: like in main.py, top_k is the number of predictions to generate

top_k = st.number_input( # this is actually not a text box but instead a slider/widget
    label = "Enter the amount of top predictions from the model that you would like to display",
    min_value = 1,
    max_value = 20,
    value = 5, #default value
    step = 1,
    help = "how many predictions should the model return? From 1 to 20"
    
)

if st.button("Predict", type = "primary"):
    with st.spinner("Model running..."):
        image = Image.open(selected_image).convert("RGB")
        image_tensor = preprocess_image(image)
        predictions = predict(model, image_tensor, labels, top_k = int(top_k))

    st.success("Prediction complete successfully")
    st.markdown(f"TOP {top_k} PREDICTIONS:")
    for rank, (class_name, confidence) in enumerate(predictions, start = 1):
        st.markdown(f"**{rank}.** {class_name}")
        st.progress(confidence / 100) # this is a progress bar which is filled to the confidence percentage for that prediction. 
        st.caption(f"{confidence:.2f}%")

st.divider()
st.subheader("In the short future: the Grad-CAM heatmap will be added. Then, the perturbations menu will be added. Then, the analysis/choose example outputs features will be added. This will be a polished app very soon, so stay in touch with this project.")
feedback = st.text_input("Questions? Email me at aarusharya@berkeley.edu")


# now display predictions. took me a time to handle edge cases where an image was actually not selected by the user. 
# but while doing so, i learned a lot from the streamlit api. even adder a spinner. I'll probably work on polishing this UI at the very end of this project




# note for the future

# grad cam heatmap should be displayed nicely in a box in the middle of the screen (need to figure out streamlit), ABOVE the display of the original selected image. a nice label to seperate these two.
# instead of select an image, there will be a completely other option (highest layer) to instead view example outputs


# add more to the examples menu, standardize this project and the file structure (at the end, this is a later step)

    
         


    



