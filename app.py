
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
    " INSTRUCTIONS and BACKGROUND (WHAT THIS IS) TO BE INSERTED HERE. "
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

        # NOTE: Streamlit "help" feature for sliders displays a question box near the slider, showing an explanation upon hover.
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
        if use_rotation:
            rotation_angle = st.slider(
                "Rotation Angle (in degrees counterclockwise from 1 to 180 degrees)",
                min_value = 1, max_value = 180, value = 30, step = 1,
                help = "This is the amount of degrees to rotate the input image counterclockwise"
            )
        else:
            rotation_angle = 0

        use_occlusion = st.checkbox("Occlusion")
        if use_occlusion:
            occlusion_size = st.slider(
                "Size of the black occlusion box in pixels.",
                min_value = 10, max_value = 150, value = 64, step = 2,
                help = "Width and height of the black occlusion rectangle to go on the image (location determined by the below controls)"
            )

            # NOTE: fix later so it adapts to image size... :(

            occlusion_x = st.slider(
                "Distance of the left edge of the occlusion rectangle from the left edge of the image, in pixels. Default is centered",
                min_value = 0, max_value = 180, value = 50,
                help = "Left edge of the occlusion rectangle. Default is centered"
            )

            occlusion_y = st.slider(
                "Distance of the top edge of the occlusion rectangle from the top edge of the image, in pixels. Default is centered",
                min_value = 0, max_value = 180, value = 50,
                help = "Top edge of the occlusion rectangle. Default is centered"
            )
        else:
            occlusion_size = 0
            occlusion_x = 0
            occlusion_y = 0
        
    # the following is used later when the output grid is arranged in the streamlit app
    any_perturbation = use_noise or use_rotation or use_occlusion
    

    st.divider()
    
    predict_clicked = st.button("Run (Predict and display heatmap)", type = "primary")


    # now display predictions and grad cam. took me a time to handle edge cases where an image was actually not selected by the user. 
    # but while doing so, i learned a lot from the streamlit api. even adder a spinner. I'll probably work on polishing this UI at the very end of this project
    if predict_clicked:
        with st.spinner("Model running...Grad-CAM generating (with all applied perturbation(s))..."): # for future: Perturbated Grad-CAM generating...
            original_image = Image.open(selected_image).convert("RGB")
            original_tensor = preprocess_image(original_image)
            original_predictions = predict(model, original_tensor, labels, top_k = int(top_k))
            original_heatmap = generate_gradcam(model, original_tensor) # calling grad cam methods
            original_overlaid = overlay_heatmap_on_image(original_image, original_heatmap)


            if any_perturbation:
                # time to apply all the perturbations one at a time. very intuitive
                perturbed_image = original_image.copy()
                if use_noise:
                    perturbed_image = add_gaussian_noise(perturbed_image, strength = noise_strength)
                if use_rotation:
                    perturbed_image = add_rotation(perturbed_image, angle = rotation_angle)
                if use_occlusion:
                    perturbed_image = add_occlusion(perturbed_image, width = occlusion_size, height = occlusion_size, x = occlusion_x, y = occlusion_y)
                perturbed_tensor = preprocess_image(perturbed_image)
                # the new predictions
                perturbed_predictions = predict(model, perturbed_tensor, labels, top_k = int(top_k))
                perturbed_heatmap = generate_gradcam(model, perturbed_tensor) 
                perturbed_overlaid = overlay_heatmap_on_image(perturbed_image, perturbed_heatmap)


        st.divider()

        # NOTE: Initially, I was simply displaying all images below the last (i.e. the grad-cam)
        # result was right below the display of the picture that the user chose. I am going to change this.
        # With the addition of the perturbations, the results will be displayed in a 2x2 Grid.
        # - With top left being chosen image
        # - bottom left being original image with all perturbations applied
        # - top right being Grad-CAM for image with no perturbations
        # - bottom right being Grad-CAM for image with all perturbations.

        # In the case that no perturbations are applied, the layout is a simple two column layout
        # the 0.05 is a space between the two side by side columns.
        if not any_perturbation:
            col_original, col_spacer, col_heatmap = st.columns([1, 0.05, 1]) # original image, spacing, heatmap respectively

            with col_original:
                st.markdown("YOUR ORIGINAL IMAGE")
                st.image(original_image, use_container_width=True)
 
            with col_heatmap:
                st.markdown("GENERATED Grad-CAM HEATMAP")
                st.image(original_overlaid, use_container_width=True)
                st.header("What does the Grad-CAM mean?")
                st.caption(
                    "Red regions are what most influenced the model's prediction."
                    "Blue regions had the LEAST influence on the model's prediction."
                    "To learn more about how Grad-CAM (or the rest of this project) works, see References or Background from the menu bar" # for future implementation btw 
                )
        
        # the 2x2 display grid 
        else:

            st.markdown("RESULTS (2X2 GRID)")
            st.caption("The top row are the images (original and with perturbations). The bottom row are the respective Grad-CAM heatmaps")
            
            # top part of the grid is the same code as in the if statement
            col_original, col_spacer, col_original_heatmap = st.columns([1, 0.05, 1])
            with col_original:
                st.markdown("YOUR ORIGINAL IMAGE")
                st.image(original_image, use_column_width=True)
            with col_original_heatmap:
                st.markdown("ORIGINAL (NO PERTURBATIONS) Grad-CAM HEATMAP")
                st.image(original_overlaid, use_container_width=True)
                st.header("What does the Grad-CAM mean?")
                st.caption(
                    "Red regions are what most influenced the model's prediction."
                    "Blue regions had the LEAST influence on the model's prediction."
                    "To learn more about how Grad-CAM (or the rest of this project) works, see References or Background from the menu bar" # for future implementation btw 
                )
            # bottom part of the grid
            col_new, col_spacer2, col_new_heatmap = st.columns([1, 0.05, 1])
            with col_new:
                st.markdown("IMAGE WITH ALL PERTURBATIONS APPLIED")
                st.image(perturbed_image, use_column_width=True)
            with col_new_heatmap:
                st.markdown("FINAL Grad-CAM HEATMAP (after all applied image distortions)")
                st.image(perturbed_overlaid, use_container_width=True)
                st.header("What does the Grad-CAM mean?")
                st.caption(
                    "Red regions are what most influenced the model's prediction."
                    "Blue regions had the LEAST influence on the model's prediction."
                    "To learn more about how Grad-CAM (or the rest of this project) works, see References or Background from the menu bar" # for future implementation btw 
                )

            st.divider()

            # Now for the predictions, again using the boolean any_perturbation
            # I changed the initial prediction code to use a seperate column for each prediction instead of each prediction being a markdown. This just looks cleaner and nicer
            
            if not any_perturbation:
                st.markdown(f"TOP {top_k} PREDICTIONS:")
                num_cols = min(int(top_k), 5)
                prediction_columns = st.columns(num_cols)
                for i, (class_name, confidence) in enumerate(original_predictions):
                    with prediction_columns[i % num_cols]:
                        st.markdown(f"**#{i + 1}** {class_name}")
                        st.progress(confidence / 100)
                        st.caption(f"{confidence:.2f}%")
            else:
                # Left and right columns, built the same way as the columns for the heatmaps (0.05 spacing in between)

                normal_prediction_column, prediction_space_column, perturbed_prediction_column = st.columns([1, 0.05, 1])

                with normal_prediction_column:
                    st.markdown("Original Predictions (with no perturbations applied)")
                    # Note: writing the prediction display in the same way as it was initially, if it looks messy, il change it to the column code thats in the if not any_perturbation if statement
                    for i, (class_name, confidence) in enumerate(original_predictions):
                        st.markdown(f"**#{i + 1}** {class_name}")
                        st.progress(confidence / 100)
                        st.caption(f"{confidence:.2f}%")

                
                with perturbed_prediction_column:

                    for i, (class_name, confidence) in enumerate(perturbed_predictions):
                        st.markdown(f"**#{i + 1}** {class_name}")
                        st.progress(confidence / 100)
                        st.caption(f"{confidence:.2f}%")

            


st.divider()
st.subheader("MORE FEATURES COMING SOON! Most notably, this will include an option to choose and analyze key example outputs instead of selecting your own image. A download feature will also be added. I also hope to add significantly more labels to feed into the model for a greater and more realistic variety for classification. This will be a polished app very soon, so STAY IN TOUCH with this project.")
feedback = st.text_input("Questions? Email me at aarusharya@berkeley.edu")







# note for the future


# instead of select an image, there will be a completely other option (highest layer) to instead view example outputs

# need a download result feature

# add more to the examples menu, standardize this project and the file structure (at the end, this is a later step)
# and format the "or choose one of our images" and  "choose an example output" to look nice -- like those images are actually there. so its visually appealing. will need to find a way to do this later on.

# oh yeah also should have a menu bar with other options like references, background, about (basically readme.md) and need to populate

# SHOULD FEED IN MORE LABELS TO THE MODEL!!!! SO IT CAN DO OTHER, MORE COMPLEX STUFF





    



