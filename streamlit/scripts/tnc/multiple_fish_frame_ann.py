import streamlit as st
import pandas as pd
import os
from PIL import Image, ImageDraw, ImageFont

# Hardcoded paths
CSV_FILE_PATH = (
    "../data/tnc/set1/csv/combind-sizes-v2.csv"  # Replace with the actual path
)
BASE_DIR = "/mnt/d/project-monocular-depth/projects-v1/tnc-ground-truth/data/"  # Replace with the actual base directory


# Load the CSV file
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)


# Draw all lines on the image and label them with IDs
def draw_lines_on_image(image_path, coords_list, ids_list):
    try:
        # Open the image
        img = Image.open(image_path).convert("RGBA")
        draw = ImageDraw.Draw(img)

        # If desired, you can specify a font. Make sure a .ttf font file is accessible.
        # For example, if you have a suitable font file, you could do:
        # font = ImageFont.truetype("arial.ttf", 20)
        # Otherwise, this will just use a default bitmap font.
        # font = None
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)

        # Draw all lines and their IDs
        for (Lx0, Ly0, Lx1, Ly1, inverted_line), annotation_id in zip(
            coords_list, ids_list
        ):
            inverted_line = (
                str(inverted_line).strip().lower()
                if pd.notnull(inverted_line)
                else "no"
            )
            if inverted_line == "yes":
                # Swap y-coordinates if inverted
                Ly0, Ly1 = Ly1, Ly0

            # Draw the line
            draw.line([(Lx0, Ly0), (Lx1, Ly1)], fill="red", width=5)

            # Choose a position for the text label. Let's put it at the midpoint of the line.
            midpoint_x = (Lx0 + Lx1) / 2
            midpoint_y = (Ly0 + Ly1) / 2

            # Draw the ID text near the line
            # Using a white fill for visibility and possibly adding a contrasting outline
            draw.text(
                (midpoint_x, midpoint_y), str(annotation_id), fill="white", font=font
            )

        return img
    except Exception as e:
        st.error(f"Error drawing lines on image: {e}")
        return None


def main():
    st.title("Duplicate Frame Viewer with Object Ranges")

    # Load the hardcoded CSV file
    try:
        data = load_data(CSV_FILE_PATH)
    except FileNotFoundError:
        st.error(f"CSV file not found at: {CSV_FILE_PATH}")
        return
    except Exception as e:
        st.error(f"An error occurred while loading the CSV file: {e}")
        return

    # Filter out rows where sizes == "sizes01"
    data = data[data["sizes"] != "sizes01"]

    # Filter to include only rows where camera == "L"
    data = data[data["camera"] == "L"]

    # Validate required columns
    required_columns = [
        "frame",
        "deployment-code",
        "calibration-id",
        "sizes",
        "Lx0",
        "Ly0",
        "Lx1",
        "Ly1",
        "inverted_line",
        "camera",
        "object-range",
        "id",  # Make sure "id" is indeed a column in your CSV
    ]
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        st.error(f"Missing columns in the CSV file: {missing_columns}")
        return

    # Group by the relevant columns and find duplicates
    grouped = data.groupby(["deployment-code", "calibration-id", "sizes", "frame"])
    duplicate_counts = grouped.size().reset_index(name="count")
    duplicates = duplicate_counts[duplicate_counts["count"] > 1]

    if duplicates.empty:
        st.info("No duplicate frames found.")
        return

    # State management for navigation
    if "index" not in st.session_state:
        st.session_state.index = 0

    # Get the current duplicate row
    current_row = duplicates.iloc[st.session_state.index]

    # Extract details for current duplicate
    size_folder = current_row["sizes"]
    frame = current_row["frame"]
    deployment = current_row["deployment-code"]
    calibration = current_row["calibration-id"]

    # Get all line coordinates, object ranges, and IDs for the current duplicate
    annotations = data[
        (data["sizes"] == size_folder)
        & (data["frame"] == frame)
        & (data["deployment-code"] == deployment)
        & (data["calibration-id"] == calibration)
    ][["Lx0", "Ly0", "Lx1", "Ly1", "inverted_line", "object-range", "id"]]

    coords_list = annotations[
        ["Lx0", "Ly0", "Lx1", "Ly1", "inverted_line"]
    ].values.tolist()
    object_ranges = annotations["object-range"].values.tolist()
    ids_list = annotations["id"].values.tolist()

    # Construct the image path
    image_path = os.path.join(
        BASE_DIR,
        size_folder,
        "input-frames",
        "left",
        f"{deployment}_{calibration}_L_{frame}.jpg",
    )

    # Display the image and details
    st.subheader(
        f"Frame: {frame} | Deployment: {deployment} | Calibration: {calibration}"
    )
    st.text(f"Size Folder: {size_folder}")

    if os.path.exists(image_path):
        annotated_img = draw_lines_on_image(image_path, coords_list, ids_list)
        if annotated_img:
            st.image(
                annotated_img, caption=f"Frame: {frame} with Line Annotations and IDs"
            )
    else:
        st.warning(f"Image not found: {image_path}")

    # Sidebar for object ranges
    st.sidebar.title("Object Ranges")
    st.sidebar.subheader(f"Frame: {frame}")

    # Instead of using enumerate, zip the ids_list and object_ranges together
    for annotation_id, obj_range in zip(ids_list, object_ranges):
        st.sidebar.markdown(f"**Annotation {annotation_id}:** {obj_range}")

    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous"):
            st.session_state.index = max(0, st.session_state.index - 1)
        st.text("")  # Add space for alignment
    with col2:
        if st.button("Next"):
            st.session_state.index = min(
                len(duplicates) - 1, st.session_state.index + 1
            )

    st.text(f"Viewing {st.session_state.index + 1} of {len(duplicates)} duplicates.")


if __name__ == "__main__":
    main()
