"""
Streamlit Application for Fish Data Annotation and Comparison

This app allows users to load fish data from a CSV file and view annotated images
based on user-selected parameters + ranks.

1. Load data from a CSV file.
2. Filter data based on user selections (size, camera, deployment code, frame).
3. Construct and retrieve image paths for the selected images.
4. Draw lines and rankings on images based on coordinates and rankings provided in the data.
5. Display the original ground truth and predicted images side by side with annotations.

The images are displayed with lines drawn between specified coordinates, and rankings are
shown at the starting point of each line.
"""

import streamlit as st
import pandas as pd
import os
from PIL import Image, ImageDraw, ImageFont

# Path to the CSV file and base image directory
CSV_PATH = "/home/shakyafernando/projects/monocular-depth/home/ubuntu/stereo-app-tnc/data/ranks-fl/ranks00-nojade-original.csv"
BASE_IMAGE_DIR = "/home/shakyafernando/projects/monocular-depth/frames"


# Function to load data
@st.cache_data
def load_data(file_path):
    """
    Load data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    df = pd.read_csv(file_path)
    return df


# Function to filter data
def filter_data(df, column, filter_value):
    """
    Filter the DataFrame based on a specified column and filter value.

    Args:
        df (pd.DataFrame): DataFrame to filter.
        column (str): The column name to filter by.
        filter_value (str or int): The value to filter by.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    return df[df[column] == filter_value]


# Function to construct image filenames
def construct_filenames(deployment_code, calibration_id, camera, frame):
    """
    Construct filenames for the main, predicted outdoor, and predicted indoor images.

    Args:
        deployment_code (str): The deployment code.
        calibration_id (str): The calibration ID.
        camera (str): The camera type ('L' or 'R').
        frame (int): The frame number.

    Returns:
        tuple: Filenames for the main image, predicted outdoor image, and predicted indoor image.
    """
    base_filename = f"{deployment_code}_{calibration_id}_{camera}_{frame}"
    main_filename = f"{base_filename}.jpg"
    pred_outdoor_filename = f"{base_filename}_pred04.png"
    pred_indoor_filename = f"{base_filename}_pred04.png"
    return main_filename, pred_outdoor_filename, pred_indoor_filename


# Function to get the image path
def get_image_path(base_dir, size, sub_dir, filename):
    """
    Construct the full path to an image.

    Args:
        base_dir (str): Base directory of the images.
        size (str): Size folder.
        sub_dir (str): Sub-directory within the base directory.
        filename (str): Image filename.

    Returns:
        str: Full path to the image.
    """
    return os.path.join(base_dir, size, sub_dir, filename)


# Function to draw lines and ranks on images based on coordinates
def draw_lines_and_ranks_on_image(image_path, coords_list, ranks, inverted_line):
    """
    Draw lines and rankings on an image based on provided coordinates and rankings.

    Args:
        image_path (str): Path to the image.
        coords_list (list of tuples): List of coordinates (x0, y0, x1, y1) for the lines.
        ranks (list): List of ranks corresponding to each set of coordinates.
        inverted_line (str): Indicates if the line coordinates should be inverted.

    Returns:
        PIL.Image.Image: Image with lines and rankings drawn, or None if image does not exist.
    """
    if not os.path.exists(image_path):
        return None

    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    for coords, rank in zip(coords_list, ranks):
        x0, y0, x1, y1 = coords
        if inverted_line.lower() == "yes":
            y0, y1 = y1, y0
        draw.line([(x0, y0), (x1, y1)], fill="red", width=2)
        draw.text((x0, y0), str(rank), fill="blue")

    return image


# Streamlit app layout
st.set_page_config(page_title="Fish Ranking Comparison", layout="wide")

# Load data
df = load_data(CSV_PATH)

# Sidebar for image options
st.sidebar.title("Image Filter Options")
size = st.sidebar.selectbox("Select Size", ["size00", "size01", "size02", "size03"])
sub_dir = st.sidebar.selectbox("Select Input Camera", ["Left", "Right"])
sub_dir = "input-frames/left" if sub_dir == "Left" else "input-frames/right"
camera = "L" if "left" in sub_dir else "R"

deployment_code = st.sidebar.selectbox(
    "Select Deployment Code", df["deployment-code"].unique()
)

calibration_id = df[df["deployment-code"] == deployment_code]["calibration-id"].iloc[0]
st.sidebar.write(f"Calibration ID: **{calibration_id}**")
frame = st.sidebar.text_input("Enter Frame", "")

if st.sidebar.button("Display"):
    if deployment_code and frame.isdigit() and calibration_id and camera:
        frame = int(frame)
        main_filename, pred_outdoor_filename, pred_indoor_filename = (
            construct_filenames(deployment_code, calibration_id, camera, frame)
        )

        main_image_path = get_image_path(BASE_IMAGE_DIR, size, sub_dir, main_filename)
        pred_outdoor_image_path = get_image_path(
            BASE_IMAGE_DIR,
            size,
            "predicted-outdoor-model-output/left",
            pred_outdoor_filename,
        )
        pred_indoor_image_path = get_image_path(
            BASE_IMAGE_DIR,
            size,
            "predicted-indoor-model-output/left",
            pred_indoor_filename,
        )

        rows = df[
            (df["deployment-code"] == deployment_code)
            & (df["calibration-id"] == calibration_id)
            & (df["camera"] == camera)
            & (df["frame"] == frame)
        ]
        if not rows.empty:
            st.write("### Corresponding Data")
            st.write(rows)

            # Sort rows based on rank_object_range for ground truth
            rows_sorted = rows.sort_values(by="rank_object_range")

            # Get all coordinates and inversion info
            coords_list = [
                (row["Lx0"], row["Ly0"], row["Lx1"], row["Ly1"])
                for _, row in rows_sorted.iterrows()
            ]
            inverted_line = rows_sorted["inverted_line"].iloc[0]

            cols = st.columns(3)

            if os.path.exists(main_image_path):
                with cols[0]:
                    st.write("### Ground Truth")
                    main_image = draw_lines_and_ranks_on_image(
                        main_image_path,
                        coords_list,
                        rows_sorted["rank_object_range"].tolist(),
                        inverted_line,
                    )
                    st.image(
                        main_image,
                        caption=f"{deployment_code}_{calibration_id}_{camera}_{frame}",
                        use_column_width=True,
                    )
                    st.write(
                        f"**Ground Truth Ranking:** {list(rows_sorted['rank_object_range'])}"
                    )
            else:
                with cols[0]:
                    st.write(
                        f"No ground truth image found for {deployment_code}_{calibration_id}_{camera}_{frame}"
                    )

            if os.path.exists(pred_indoor_image_path):
                with cols[1]:
                    st.write("### Indoor Prediction")
                    pred_indoor_image = draw_lines_and_ranks_on_image(
                        pred_indoor_image_path,
                        coords_list,
                        rows_sorted["rank_value_indoor_left"].tolist(),
                        inverted_line,
                    )
                    st.image(
                        pred_indoor_image,
                        caption=f"{deployment_code}_{calibration_id}_{camera}_{frame}_pred04",
                        use_column_width=True,
                    )
                    st.write(
                        f"**Indoor Prediction Ranking:** {list(rows_sorted['rank_value_indoor_left'])}"
                    )
            else:
                with cols[1]:
                    st.write(
                        f"No indoor prediction image found for {deployment_code}_{calibration_id}_{camera}_{frame}_pred04"
                    )

            if os.path.exists(pred_outdoor_image_path):
                with cols[2]:
                    st.write("### Outdoor Prediction")
                    pred_outdoor_image = draw_lines_and_ranks_on_image(
                        pred_outdoor_image_path,
                        coords_list,
                        rows_sorted["rank_value_outdoor_left"].tolist(),
                        inverted_line,
                    )
                    st.image(
                        pred_outdoor_image,
                        caption=f"{deployment_code}_{calibration_id}_{camera}_{frame}_pred04",
                        use_column_width=True,
                    )
                    st.write(
                        f"**Outdoor Prediction Ranking:** {list(rows_sorted['rank_value_outdoor_left'])}"
                    )
            else:
                with cols[2]:
                    st.write(
                        f"No outdoor prediction image found for {deployment_code}_{calibration_id}_{camera}_{frame}_pred04"
                    )

        else:
            st.write(
                "No corresponding data found in the DataFrame for the provided inputs."
            )
    else:
        st.write("Please enter valid Deployment Code and Frame.")
