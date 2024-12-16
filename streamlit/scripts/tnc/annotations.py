"""
Streamlit Application for Viewing Annotated Fish Data

This application loads data from a CSV file and allows users to filter the data by 
size, user, and deployment code. Users can view annotated images (line) from the left and 
right cameras side by side, with the option to navigate through the frames in a sequence. 
The application also counts and displays the number of sequences within the selected data 
and indicates the current sequence being viewed.

"""

import streamlit as st
import pandas as pd
import os
from PIL import Image, ImageDraw

# Path to the CSV file and base image directory
CSV_PATH = "/home/shakyafernando/projects/monocular-depth/home/ubuntu/stereo-app-tnc/data/combind-sizes-v1.csv"
BASE_IMAGE_DIR = "/home/shakyafernando/projects/monocular-depth/data/frames"


def load_data(file_path):
    """
    Load data from a CSV file and preprocess the columns.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Preprocessed data as a DataFrame.
    """
    df = pd.read_csv(file_path)
    df["deployment-code"] = df["deployment-code"]
    df["calibration-id"] = df["calibration-id"].astype(str)
    df["camera"] = df["camera"].astype(str)
    df["frame"] = df["frame"].astype(int)
    df["user"] = df["user"].astype(str)
    return df


def reset_session_state():
    """
    Reset the session state for frame navigation.
    """
    st.session_state.current_frame_index = 0


def get_image_path(base_dir, size, sub_dir, filename):
    """
    Construct the full image path.

    Args:
        base_dir (str): Base directory of the images.
        size (str): Size folder.
        sub_dir (str): Sub-directory within the base directory.
        filename (str): Image filename.

    Returns:
        str: Full path to the image.
    """
    return os.path.join(base_dir, size, sub_dir, filename)


def construct_filenames(deployment_code, calibration_id, camera, frame):
    """
    Construct the image filenames based on deployment code, calibration ID, camera, and frame number.

    Args:
        deployment_code (str): Deployment code.
        calibration_id (str): Calibration ID.
        camera (str): Camera type ('L' for left, 'R' for right).
        frame (int): Frame number.

    Returns:
        str: Image filename.
    """
    base_filename = f"{deployment_code}_{calibration_id}_{camera}_{frame}"
    main_filename = f"{base_filename}.jpg"
    return main_filename


def swap_y_if_inverted(y0, y1, inverted):
    """
    Swaps y-coordinates if the line is inverted.

    Args:
        y0 (int): The y-coordinate of the first point.
        y1 (int): The y-coordinate of the second point.
        inverted (bool): Indicates if the line is inverted.

    Returns:
        tuple: The possibly swapped y-coordinates.
    """
    if inverted:
        y0, y1 = y1, y0
    return y0, y1


def draw_line_on_image(image, coords, inverted=False, color="red", width=2):
    """
    Draw a line on an image based on provided coordinates and inversion status.

    Args:
        image (PIL.Image.Image): The image to draw on.
        coords (tuple): Tuple of coordinates (x0, y0, x1, y1).
        inverted (bool): Indicates if the line coordinates are inverted.
        color (str): Line color.
        width (int): Line width.

    Returns:
        PIL.Image.Image: Image with the line drawn.
    """
    x0, y0, x1, y1 = coords
    if inverted:
        y0, y1 = swap_y_if_inverted(y0, y1, inverted)
    draw = ImageDraw.Draw(image)
    draw.line((x0, y0, x1, y1), fill=color, width=width)
    return image


def count_sequences_and_get_current_sequence(frames, current_index):
    """
    Count the number of sequences and identify the current sequence number.

    Args:
        frames (list): List of frame numbers.
        current_index (int): Current frame index.

    Returns:
        tuple: Total number of sequences and the current sequence number.
    """
    sequences = 1
    sequence_indices = [0]  # Start of each sequence
    for i in range(1, len(frames)):
        if frames[i] != frames[i - 1] and frames[i] != frames[i - 1] + 1:
            sequences += 1
            sequence_indices.append(i)

    # Determine the current sequence number
    current_sequence = 1
    for i in range(1, len(sequence_indices)):
        if current_index >= sequence_indices[i]:
            current_sequence = i + 1
        else:
            break

    return sequences, current_sequence


# Streamlit app layout
st.set_page_config(page_title="Explore Fish Data", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Image Viewer"])

if page == "Image Viewer":
    st.title("Image Viewer")
    df = load_data(CSV_PATH)

    st.sidebar.title("Image Filter Options")
    st.sidebar.write(
        "Select a Size, User, and Deployment Code - then navigate through each frame."
    )
    st.sidebar.write('**NOTE:** First click of "Next Frame" loads the first frame.')

    # Selection for Size
    size_options = df["sizes"].unique().tolist()
    selected_size = st.sidebar.selectbox(
        "Select Size", size_options, on_change=reset_session_state
    )

    filtered_df = df[df["sizes"] == selected_size]

    # Selection for User
    user_options = ["All"] + filtered_df["user"].unique().tolist()
    selected_user = st.sidebar.selectbox(
        "Select User", user_options, on_change=reset_session_state
    )

    if selected_user != "All":
        filtered_df = filtered_df[filtered_df["user"] == selected_user]

    # Selection for Deployment Code
    deployment_code_options = filtered_df["deployment-code"].unique()
    deployment_code = st.sidebar.selectbox(
        "Select Deployment Code", deployment_code_options, on_change=reset_session_state
    )

    calibration_id = filtered_df[filtered_df["deployment-code"] == deployment_code][
        "calibration-id"
    ].iloc[0]
    st.sidebar.write(f"Calibration ID: **{calibration_id}**")

    # Initialize session state for frame navigation
    if "current_frame_index" not in st.session_state:
        st.session_state.current_frame_index = 0

    # Get the frames for the selected deployment code and calibration id
    frames = (
        filtered_df[
            (filtered_df["deployment-code"] == deployment_code)
            & (filtered_df["calibration-id"] == calibration_id)
        ]["frame"]
        .sort_values()
        .tolist()
    )

    # Count the number of sequences and get the current sequence
    num_sequences, current_sequence = count_sequences_and_get_current_sequence(
        frames, st.session_state.current_frame_index
    )
    st.write(f"Number of Sequences: **{num_sequences}**")
    st.write(f"Current Sequence: **{current_sequence}**")

    if len(frames) > 0:
        current_frame = frames[st.session_state.current_frame_index]
        st.sidebar.write(f"Current Frame: **{current_frame}**")

        # Get coordinates and inversion status for the current frame
        current_coords = filtered_df[
            (filtered_df["deployment-code"] == deployment_code)
            & (filtered_df["calibration-id"] == calibration_id)
            & (filtered_df["frame"] == current_frame)
        ].iloc[0]

        # Convert to string and handle possible NaN values
        inverted = str(current_coords["inverted_line"]).lower() == "yes"

        left_coords = (
            current_coords["Lx0"],
            current_coords["Ly0"],
            current_coords["Lx1"],
            current_coords["Ly1"],
        )

        right_coords = (
            current_coords["Rx0"],
            current_coords["Ry0"],
            current_coords["Rx1"],
            current_coords["Ry1"],
        )

        # Construct image filenames and paths
        left_filename = construct_filenames(
            deployment_code, calibration_id, "L", current_frame
        )
        right_filename = construct_filenames(
            deployment_code, calibration_id, "R", current_frame
        )

        left_image_path = get_image_path(
            BASE_IMAGE_DIR, selected_size, "input-frames/left", left_filename
        )
        right_image_path = get_image_path(
            BASE_IMAGE_DIR, selected_size, "input-frames/right", right_filename
        )

        # Display Images
        cols = st.columns(2)

        with cols[0]:
            if os.path.exists(left_image_path):
                st.header("**Left Camera:**")
                left_image = Image.open(left_image_path)
                left_image = draw_line_on_image(
                    left_image, left_coords, inverted=inverted
                )
                st.image(
                    left_image,
                    caption=f"Left Image: {deployment_code}_{calibration_id}_L_{current_frame}",
                    use_column_width=True,
                )
            else:
                st.write(
                    f"No left image found for {deployment_code}_{calibration_id}_L_{current_frame}"
                )

        with cols[1]:
            if os.path.exists(right_image_path):
                st.header("**Right Camera:**")
                right_image = Image.open(right_image_path)
                right_image = draw_line_on_image(
                    right_image, right_coords, inverted=inverted
                )
                st.image(
                    right_image,
                    caption=f"Right Image: {deployment_code}_{calibration_id}_R_{current_frame}",
                    use_column_width=True,
                )
            else:
                st.write(
                    f"No right image found for {deployment_code}_{calibration_id}_R_{current_frame}"
                )

        # Create a row with the Previous and Next buttons
        button_cols = st.sidebar.columns(2)

        with button_cols[0]:
            if st.button(
                "Previous Frame", disabled=st.session_state.current_frame_index == 0
            ):
                if st.session_state.current_frame_index > 0:
                    st.session_state.current_frame_index -= 1

        with button_cols[1]:
            if st.button(
                "Next Frame",
                disabled=st.session_state.current_frame_index >= len(frames) - 1,
            ):
                if st.session_state.current_frame_index < len(frames) - 1:
                    st.session_state.current_frame_index += 1

    else:
        st.write("No frames available for the selected filters.")
