
import streamlit as st
import os

# NOTE: DO NOT uncomment the imports! these are imported conditionally later in the file.

#from model_ResNet50 import load_model, load_labels, preprocess_image, predict, get_gradcam_layer
#from model_VGG16 import load_model, load_labels, preprocess_image, predict, get_gradcam_layer
#from model_EfficientNetB0 import load_model, load_labels, preprocess_image, predict, get_gradcam_layer
from PIL import Image 

from gradcam import generate_gradcam, overlay_heatmap_on_image

from perturbations import add_gaussian_noise, add_rotation, add_occlusion



# HOW TO RUN:
# python -m streamlit run app.py

# project two-line summary: 
# An interactive platform for exploring CNN interpretability and robustness. 
# Users can compare Grad-CAM explanations across multiple architectures and evaluate prediction stability under perturbations such as Gaussian noise, rotation, and occlusion.
st.set_page_config(
    page_title = "Interactive CNN Visualizer",
    page_icon = "🤖",
    layout = "wide" # to use the full browser width
)


st.title("Interactive CNN Visualizer")

# was initially having all of this text below the title, but it is nicer to have buttons and popups for a cleaner UI
instructions_column, background_column, spacing_column = st.columns([1, 1, 4])

with instructions_column:
    if st.button("How to Use"):
        @st.dialog("How to Use")
        def show_instructions():
            with open("instructions.md", "r") as f:
                st.markdown(f.read()) # see the file instructions.md
        show_instructions()

    with background_column:
        if st.button("Background, References, & Tools Used"):
            @st.dialog("Background, References, & Tools Used")
            def show_background():
                with open("background_references.md", "r") as f:
                    st.markdown(f.read()) # see the file background_references.md
            show_background()


st.divider()

st.subheader("Step 1: Select a Model Architecture")
model_choice = st.selectbox(
    "Choose one of three vastly different model architectures: ResNet-50, EfficientNet-B0, and VGG-16.",
    ["ResNet-50", "VGG-16", "EfficientNet-B0"],
    help="ResNet-50: Uses skip connections for easy gradient flow in the network. Reliable image classification and this project's baseline. VGG-16: A classic CNN with a more sequential structure, which leads to different attention patterns. EfficientNet-B0: Balances depth, width and resolution to extract features as efficiently as possible."
)
st.caption("Different architectures fundamentally may lead to different predictions and produce different Grad-CAM attention patterns, especially when subject to varying image complexity and varying perturbation levels (select perturbations in Step 3.)")
st.caption('Click the "?" icon in the corner above the dropdown to learn more.')
st.caption("For further background on these model architectures and how they differ, see Background, References, & Tools Used.")


# Different model architectures implemented via selectbox, so we need to account for that in get_model and get_labels by passing in the model as a parameter later on

# to avoid issues with the imports, probably best to just import whatever aligns with the chosen model.
# decided to just have a seperate method to do this although probably won't need it more than once?
def get_model_functions(architecture):
    if architecture == "ResNet-50":
        from model_ResNet50 import load_model, load_labels, preprocess_image, predict, get_gradcam_layer
    elif architecture == "VGG-16":
        from model_VGG16 import load_model, load_labels, preprocess_image, predict, get_gradcam_layer
    elif architecture == "EfficientNet-B0":
        from model_EfficientNetB0 import load_model, load_labels, preprocess_image, predict, get_gradcam_layer
    else:
        print(f"Unknown Architecture {architecture} passed through??")

    return load_model, load_labels, preprocess_image, predict, get_gradcam_layer

load_model, load_labels, preprocess_image, predict, get_gradcam_layer = get_model_functions(model_choice)

@st.cache_resource
def get_model(architecture):
    load_model, _, _, _, _ = get_model_functions(architecture)
    return load_model()
 
@st.cache_resource
def get_labels(architecture):
    _, load_labels, _, _, _ = get_model_functions(architecture)
    return load_labels()
 
model = get_model(model_choice)
labels = get_labels(model_choice)
# Now, model and labels are set correctly

st.subheader("Step 2: Select an Image: follow the steps below")

input_method = st.radio(
    label = "Choose an image in one of the following ways: ",
    options = ["Upload your own image", "Choose an example image (use the dropdown)"],
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



if selected_image is not None:
    st.divider()
   
    st.image(selected_image, width=300, caption="Selected image") # display the selected image
    st.subheader("Step 3: Configure the settings of your trial. Use the controls below.")

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
        st.caption("Current options include Gaussian Noise (apply noise of a certain level to each pixel), Rotation (of the image by certain degrees), and Occlusions (a black square of your configuration is added to the image, hiding a certain part of the image). ")
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
            
            
            # NOTE: at first I had a set static slider for the values for occlusion_size, occlusion_x, and occlusion_y. however it became clear
            # that this causes problems since it isn't dependant on the actual image size uploaded by the user, and therefore the user can
            # easily select parameters that are out of bounds of their selected image. So instead, I below use the actual image dimensions to calculate 
            # the min, max, and default values for the occlusion parameters. 

            original_copy = Image.open(selected_image).convert("RGB")
            width, height = original_copy.size

            st.markdown(f"Note for settings: Your original image has width of {width} pixels and height of {height} pixels.")

            occlusion_size = st.slider(
                "Size (width/height) of the black occlusion box in pixels. Default is the image width over 5.",
                min_value = 5, max_value = width - 5, value = width // 5, step = 2,
                help = "Width and height of the black occlusion square to go on the image (location determined by the below controls)"
            )

            occlusion_x = st.slider(
                "Distance of the left edge of the occlusion square from the left edge of the image, in pixels. Default is centered",
                min_value = 0, max_value = width - 1, value = width // 2 - occlusion_size // 2,
                help = "Left edge of the occlusion square. Default is centered"
            )

            occlusion_y = st.slider(
                "Distance of the top edge of the occlusion square from the top edge of the image, in pixels. Default is centered",
                min_value = 0, max_value = height - 1, value = height // 2 - occlusion_size // 2,
                help = "Top edge of the occlusion square. Default is centered"
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
    
    if predict_clicked:
        with st.spinner("Model running...Grad-CAM generating (with all applied perturbation(s))..."): # for future: Perturbated Grad-CAM generating...
            original_image = Image.open(selected_image).convert("RGB")
            original_tensor = preprocess_image(original_image)
            original_predictions = predict(model, original_tensor, labels, top_k = int(top_k))
            target_layer = get_gradcam_layer(model)
            original_heatmap = generate_gradcam(model, original_tensor, target_layer) # calling grad cam methods
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
                target_layer = get_gradcam_layer(model)
                perturbed_heatmap = generate_gradcam(model, perturbed_tensor, target_layer) 
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

    
        
        # An issue I noticed was the image outputs were in different sizes, due to the overlaid images always being resized to 224x224 while the original images retain their original size. 
        # I first tried resizing all images to a fixed size but that led to the original images becoming very blurry
        # So instead, I resize the overlaid images to the original image size
        # fromarray is needed as original_overlaid is a numpy array, not a PIL image
        native_size = original_image.size
        display_original_overlaid = Image.fromarray(original_overlaid).resize(native_size, Image.LANCZOS)

        if not any_perturbation:
            st.markdown("### Results")
            st.caption(f"Selected Model Architecture: {model_choice}")
            st.caption("No Perturbations Applied")
            st.caption("The original image is on the left, the Grad-CAM heatmap is on the right.")
            st.caption("The model's predictions can be found below the heatmaps.")
            
            col_original, col_spacer, col_heatmap = st.columns([1, 0.05, 1]) # original image, spacing, heatmap respectively

            with col_original:
                st.markdown("Original Image")
                st.image(original_image, use_container_width=True)
 
            with col_heatmap:
                st.markdown("Generated Grad-CAM Heatmap")
                st.image(display_original_overlaid, use_container_width=True)
                
    
        # the 2x2 display grid 
        else:

            st.subheader("Results")
            st.caption(f"Selected Model Architecture: {model_choice}")
            perturbation_summary = [] # this is to tell the user what perturbations they applied. 
            if use_noise:
                perturbation_summary.append(f"Gaussian Noise (strength={noise_strength})")
            if use_rotation:
                perturbation_summary.append(f"Rotation ({rotation_angle}°)")
            if use_occlusion:
                perturbation_summary.append(f"Occlusion ({occlusion_size}×{occlusion_size}px at x={occlusion_x}, y={occlusion_y})")

            st.caption(f"Perturbations applied: {', '.join(perturbation_summary)}")
            st.caption("The top row contains the images (original and with perturbations). The bottom row are the respective Grad-CAM heatmaps.")
            st.caption("The model's original and perturbed predictions can be found below the heatmaps. You can also see the confidence changes of the model for its original prediction after image perturbations.")
            # top part of the grid is the same code as in the if statement
            col_original, col_spacer, col_original_heatmap = st.columns([1, 0.05, 1])
            with col_original:
                st.markdown("Original Image")
                st.image(original_image, use_container_width=True)
            with col_original_heatmap:
                st.markdown("Original (No Perturbations) Grad-CAM Heatmap")
                st.image(display_original_overlaid, use_container_width=True)
                
            # bottom part of the grid
            
            
            display_perturbed_overlaid = Image.fromarray(perturbed_overlaid).resize(native_size)
            col_new, col_spacer2, col_new_heatmap = st.columns([1, 0.05, 1])
            with col_new:
                st.markdown("Perturbed Image")
                st.image(perturbed_image, use_container_width=True)
            with col_new_heatmap:
                st.markdown("Final Perturbed Grad-CAM Heatmap")
                st.image(display_perturbed_overlaid, use_container_width=True)
                

            

            
        st.markdown("What does the Grad-CAM mean?")
        st.caption(
            "Red regions are what most influenced the model's prediction."
            " Blue regions had the least influence on the model's prediction."
            " To learn more about how Grad-CAM (or the rest of this project) works, see References or Background from the menu bar" # for future implementation btw 
        )
        
        st.divider()

        # Now for the predictions, again using the boolean any_perturbation
        # I changed the initial prediction code to use a seperate column for each prediction instead of each prediction being a markdown. This just looks cleaner and nicer
        if not any_perturbation:
            st.markdown(f"TOP {top_k} PREDICTIONS:")
            st.caption("The percentages from 1 to 100 indicate the model's CONFIDENCE in that particular prediction.")
            num_cols = min(int(top_k), 5)
            prediction_columns = st.columns(num_cols)
            for i, (class_name, confidence) in enumerate(original_predictions):
                with prediction_columns[i % num_cols]:
                    st.markdown(f"**#{i + 1}** {class_name}")
                    st.caption(f"{confidence:.2f}%")
        else:

            # Columns, build similarly to the columns for the heatmaps (0.05 spacing in between)
            # NOTE:decided to also calculate confidence drops, which states how much the model's confidence decreased (in percent)
            # for the original prediction after the perturbations were applied. 

            normal_prediction_column, prediction_space_column, perturbed_prediction_column, drop_space_column, confidence_drop_column = st.columns([1, 0.05, 1, 0.05, 0.6])

            with normal_prediction_column:
                st.markdown("Original Predictions (with no perturbations applied)")
                st.caption("The percentages from 1 to 100 indicate the model's CONFIDENCE in that particular prediction.")
                # Note: writing the prediction display in the same way as it was initially, if it looks messy, il change it to the column code thats in the if not any_perturbation if statement
                for i, (class_name, confidence) in enumerate(original_predictions):
                    st.markdown(f"**#{i + 1}** {class_name}")
                    st.caption(f"{confidence:.2f}%")

            
            with perturbed_prediction_column:
                st.markdown("Modified Predictions (after applied perturbations)")
                st.caption("The percentages from 1 to 100 indicate the model's CONFIDENCE in that particular prediction.")
                for i, (class_name, confidence) in enumerate(perturbed_predictions):
                    st.markdown(f"**#{i + 1}** {class_name}")
                    st.caption(f"{confidence:.2f}%")


            # displaying CONFIDENCE DROPS for each prediction
            confidence_drops = []

            # looping through both the original and perturbed predictions to get all confidence drops and append it 
            for (original_class, original_confidence), (perturbation_class, perturbation_confidence) in zip(original_predictions, perturbed_predictions):
                drop = perturbation_confidence - original_confidence
                confidence_drops.append((original_class, perturbation_class, original_confidence, perturbation_confidence, drop))
            
            with confidence_drop_column:
                st.markdown("Confidence Change (In Percentages)")
                st.caption("Change in confidence for INITIAL PREDICTION after perturbations. ")
                z = 0
                for orig_class, pert_class, orig_conf, pert_conf, drop in confidence_drops:
                    
                    st.markdown(" ")
                    # different colors for positive change and negative change, just for visual appeal
                    # above: never mind, can't change colors with streamlit markdown so instead just using emojis :)
                    # above: can use HTML instead
                    if drop < 0:
                        st.markdown(f"**#{z + 1} <span style='color:red'>{drop:+.2f}%</span>**", unsafe_allow_html=True)
                    elif drop > 0:
                        st.markdown(f"**#{z + 1} <span style='color:green'>{drop:+.2f}%</span>**", unsafe_allow_html=True)
                    else:
                        st.markdown(f"**#{z + 1} <span style='color:gray'>0.00%</span>**", unsafe_allow_html=True)
                    
                    st.caption(" ")
                    z = z + 1
        
        st.divider()

        # one of my favorite parts of this project is 
        st.subheader("Keep Experimenting!")
        if not any_perturbation:
            
            st.caption("Try applying perturbations next to see how the Grad-CAM heatmaps and predictions/confidence levels change!")
            st.caption("Additionally, try comparing different model architectures on the same image under identical (or no) perturbations. See if predictions remain the same, if (for trials with perturbations) prediction confidences drop more for a certain model, or if the Grad-CAM visualizations focus on different image regions.")
            st.caption("To run another experiment, go back and change any of the trial settings!")
        else:
            
            st.caption("Try to analyze differences between the two (original vs. perturbed) heatmaps, and how and why the perturbations could therefore influence the predictions, confidence drops, and/or cause potential hallucinations.")
            st.caption("Additionally, try comparing different model architectures on the same image under identical (or no) perturbations. See if predictions remain the same, if prediction confidences drop more for a certain model, or if the Grad-CAM visualizations focus on different image regions.")
            st.caption("To run another experiment, go back and change any of the trial settings!")
                    
        
        st.divider()
        
        st.caption("Please email me at aarusharya@berkeley.edu")


# note for the future

# add confidence drop to the predictions column NOTE: DONE
# add different model architectures! big step!! NOTE: DONE
# fill in how to use instructions, the background/references
# add to example menu (so there are a lot more example options)
# polish/standardize file structure
# update Readme.MD to be nice and insight-driven (showing instructions, test examples, etc)






    



