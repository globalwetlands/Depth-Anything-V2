"""
This Streamlit app allows users to explore and analyze fish data captured from monocular depth cameras. 
The app provides functionalities to:

1. View and filter CSV data.
2. View images associated with specific deployments and frames.
3. Plot various graphs for data visualization.
4. Optimize and visualize model functions for data groups.
5. Calculate and compare depth metrics.
6. Display optimized graphs using specific model functions.

The data is loaded from specified CSV files and image directories, with the ability to filter and analyze 
based on deployment codes, sizes, and other parameters.
"""

import streamlit as st
import pandas as pd
import altair as alt
import os
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw  # Make sure to import ImageDraw

# Path to the CSV file and base image directory
CSV_PATH = "/home/shakyafernando/projects/monocular-depth/home/ubuntu/stereo-app-tnc/data/combind-sizes.csv"
CSV_PATH_2 = (
    "/home/shakyafernando/projects/monocular-depth/data/csv/combined-sizes-indoor.csv"
)
BASE_IMAGE_DIR = "/home/shakyafernando/projects/monocular-depth/frames"


# Function to load data
@st.cache_data
def load_data(file_path):
    """
    Load CSV data and preprocess columns.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded and preprocessed DataFrame.
    """
    df = pd.read_csv(file_path)
    df["deployment-code"] = df["deployment-code"]
    df["calibration-id"] = df["calibration-id"].astype(str)
    df["camera"] = df["camera"].astype(str)
    df["frame"] = df["frame"].astype(int)
    return df


# Function to filter data
def filter_data(df, column, filter_value):
    """
    Filter DataFrame based on column and filter value.

    Args:
        df (pd.DataFrame): DataFrame to filter.
        column (str): Column name to filter by.
        filter_value (str, int): Value to filter.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    return df[df[column] == filter_value]


# Function to get the image path
def get_image_path(base_dir, size, sub_dir, filename):
    """
    Construct image path.

    Args:
        base_dir (str): Base directory.
        size (str): Size folder.
        sub_dir (str): Sub-directory.
        filename (str): Filename of the image.

    Returns:
        str: Full image path.
    """
    return os.path.join(base_dir, size, sub_dir, filename)


# Function to construct image filenames
def construct_filenames(deployment_code, calibration_id, camera, frame):
    """
    Construct image filenames.

    Args:
        deployment_code (int): Deployment code.
        calibration_id (str): Calibration ID.
        camera (str): Camera identifier.
        frame (int): Frame number.

    Returns:
        tuple: Filenames for main, predicted outdoor, and predicted indoor images.
    """
    base_filename = f"{deployment_code}_{calibration_id}_{camera}_{frame}"
    main_filename = f"{base_filename}.jpg"
    pred_outdoor_filename = f"{base_filename}_pred04.png"
    pred_indoor_filename = f"{base_filename}_pred04.png"
    return main_filename, pred_outdoor_filename, pred_indoor_filename


# Function to define the model functions
def model_func(x, a, b, func_type):
    """
    Define model functions.

    Args:
        x (float): Input value.
        a (float): Parameter a.
        b (float): Parameter b.
        func_type (str): Type of function.

    Returns:
        float: Computed value.
    """
    if func_type == "Function 1":
        return 1 / (a + (b * x))
    elif func_type == "Function 2":
        return 1 / ((a * x) + b)
    elif func_type == "Function 3":
        return a * (1 / x) + b


# Function to define the loss function (mean squared error)
def loss_func(params, x, y, func_type):
    """
    Calculate the loss (mean squared error).

    Args:
        params (list): Parameters [a, b].
        x (np.array): Input values.
        y (np.array): True values.
        func_type (str): Type of function.

    Returns:
        float: Mean squared error.
    """
    a, b = params
    y_pred = model_func(x, a, b, func_type)
    return np.mean((y - y_pred) ** 2)


# Function to optimize parameters for a given deployment
def optimize_parameters(group, func_type):
    """
    Optimize parameters for a given deployment group.

    Args:
        group (pd.DataFrame): Data group.
        func_type (str): Type of function.

    Returns:
        tuple: Optimized parameters (a, b).
    """
    x = group["x"]
    y = group["y"]
    initial_guess = [1, 1]
    result = minimize(loss_func, initial_guess, args=(x, y, func_type))
    a_opt, b_opt = result.x
    return a_opt, b_opt


# Function to calculate value_metric
def calculate_value_metric(depth_value, zmin, zmax):
    """
    Calculate value metric.

    Args:
        depth_value (float): Depth value.
        zmin (float): Minimum depth.
        zmax (float): Maximum depth.

    Returns:
        float: Calculated value metric.
    """
    return ((depth_value * (zmax - zmin)) + zmin) * 1000


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


def draw_lines_on_image(image_path, coords_list, inverted_line):
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
    if not os.path.exists(image_path):
        return None

    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    for coords in coords_list:
        x0, y0, x1, y1 = coords
        if inverted_line.lower() == "yes":
            y0, y1 = swap_y_if_inverted(y0, y1, inverted=True)
        draw.line([(x0, y0), (x1, y1)], fill="red", width=2)

    return image


# Streamlit app layout
st.set_page_config(page_title="Explore Fish Data", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Data Overview",
        "Image Viewer",
        "Plot Graphs",
        "Optimized Graphs",
        "Depth Calculation",
        "Function 3 Optimized Graphs",
    ],
)

if page == "Data Overview":
    st.title("CSV Data Viewer and Filter")
    df = load_data(CSV_PATH)

    st.sidebar.title("Data Filter Options")
    show_data_overview = st.sidebar.checkbox("Show Data Overview", value=True)
    if show_data_overview:
        st.write("### Data Overview - combined-sizes", df)

    filter_column = st.sidebar.selectbox("Select column to filter by", df.columns)
    if filter_column:
        unique_values = df[filter_column].unique()
        filter_value = st.sidebar.selectbox("Select value to filter", unique_values)
        if filter_value:
            filtered_data = filter_data(df, filter_column, filter_value)
            st.write("### Filtered Data", filtered_data)

# Update the "Image Viewer" section to include line drawing
elif page == "Image Viewer":
    st.title("Image Viewer")
    df = load_data(CSV_PATH)

    st.sidebar.title("Image Filter Options")
    show_data_overview = st.sidebar.checkbox("Show Data Overview", value=True)
    if show_data_overview:
        st.write("### Data Overview - combined-sizes", df)

    size = st.sidebar.selectbox("Select Size", ["size00", "size01", "size02", "size03"])
    sub_dir = st.sidebar.selectbox("Select Input Camera", ["Left", "Right"])
    sub_dir = "input-frames/left" if sub_dir == "Left" else "input-frames/right"
    camera = "L" if "left" in sub_dir else "R"

    deployment_code = st.sidebar.selectbox(
        "Select Deployment Code", df["deployment-code"].unique()
    )

    calibration_id = df[df["deployment-code"] == deployment_code][
        "calibration-id"
    ].iloc[0]
    st.sidebar.write(f"Calibration ID: **{calibration_id}**")
    frame = st.sidebar.text_input("Enter Frame", "")

    if st.sidebar.button("Display"):
        if deployment_code and frame.isdigit() and calibration_id and camera:
            frame = int(frame)
            main_filename, pred_outdoor_filename, pred_indoor_filename = (
                construct_filenames(deployment_code, calibration_id, camera, frame)
            )

            main_image_path = get_image_path(
                BASE_IMAGE_DIR, size, sub_dir, main_filename
            )
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

                # Get all coordinates and inversion info
                coords_list = [
                    (row["Lx0"], row["Ly0"], row["Lx1"], row["Ly1"])
                    for _, row in rows.iterrows()
                ]
                inverted_line = rows["inverted_line"].iloc[0]

                cols = st.columns(3)

                if os.path.exists(main_image_path):
                    with cols[0]:
                        st.write("### Ground Truth")
                        main_image = draw_lines_on_image(
                            main_image_path,
                            coords_list,
                            inverted_line,
                        )
                        st.image(
                            main_image,
                            caption=f"{deployment_code}_{calibration_id}_{camera}_{frame}",
                            use_column_width=True,
                        )
                else:
                    with cols[0]:
                        st.write(
                            f"No ground truth image found for {deployment_code}_{calibration_id}_{camera}_{frame}"
                        )

                if os.path.exists(pred_indoor_image_path):
                    with cols[1]:
                        st.write("### Indoor Prediction")
                        pred_indoor_image = draw_lines_on_image(
                            pred_indoor_image_path,
                            coords_list,
                            inverted_line,
                        )
                        st.image(
                            pred_indoor_image,
                            caption=f"{deployment_code}_{calibration_id}_{camera}_{frame}_pred04",
                            use_column_width=True,
                        )
                else:
                    with cols[1]:
                        st.write(
                            f"No indoor prediction image found for {deployment_code}_{calibration_id}_{camera}_{frame}_pred04"
                        )

                if os.path.exists(pred_outdoor_image_path):
                    with cols[2]:
                        st.write("### Outdoor Prediction")
                        pred_outdoor_image = draw_lines_on_image(
                            pred_outdoor_image_path,
                            coords_list,
                            inverted_line,
                        )
                        st.image(
                            pred_outdoor_image,
                            caption=f"{deployment_code}_{calibration_id}_{camera}_{frame}_pred04",
                            use_column_width=True,
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

elif page == "Plot Graphs":
    st.title("Plot Graphs")
    df = load_data(CSV_PATH)
    excluded_columns = [
        "Lx0",
        "Ly0",
        "Lx1",
        "Ly1",
        "Rx0",
        "Ry0",
        "Rx1",
        "Ry1",
        "filename",
        "baseline",
        "fx-left",
        "fy-left",
        "cx-left",
        "cy-left",
        "fx-right",
        "fy-right",
        "cx-right",
        "cy-right",
    ]
    available_columns = [col for col in df.columns if col not in excluded_columns]

    st.sidebar.title("Graph Filter Options")
    show_data_overview = st.sidebar.checkbox("Show Data Overview", value=True)
    if show_data_overview:
        st.write("### Data Overview - combined-sizes", df)

    x_column = st.sidebar.selectbox("Select X-axis column", available_columns)
    y_column = st.sidebar.selectbox("Select Y-axis column", available_columns)
    graph_type = st.sidebar.selectbox(
        "Select Graph Type", ["Scatter", "Bar", "Boxplot", "Pie"]
    )

    if st.sidebar.button("Display"):
        if x_column and y_column:
            st.write(f"### Plotting {x_column} vs {y_column} ({graph_type} Plot)")
            if graph_type == "Bar":
                chart = (
                    alt.Chart(df)
                    .mark_bar()
                    .encode(x=x_column, y=y_column)
                    .properties(width=300, height=600)
                )
            elif graph_type == "Boxplot":
                chart = (
                    alt.Chart(df)
                    .mark_boxplot()
                    .encode(x=x_column, y=y_column)
                    .properties(width=300, height=600)
                )
            elif graph_type == "Scatter":
                chart = (
                    alt.Chart(df)
                    .mark_point()
                    .encode(x=x_column, y=y_column)
                    .properties(width=300, height=600)
                )
            elif graph_type == "Pie":
                chart = (
                    alt.Chart(df)
                    .mark_arc()
                    .encode(
                        theta=alt.Theta(field=y_column, type="quantitative"),
                        color=alt.Color(field=x_column, type="nominal"),
                    )
                    .properties(width=300, height=600)
                )
            st.altair_chart(chart, use_container_width=True)

elif page == "Optimized Graphs":
    st.title("Optimized Graphs")
    df = load_data(CSV_PATH_2)

    st.sidebar.title("Graph Filter Options")
    show_data_overview = st.sidebar.checkbox("Show Data Overview", value=True)
    if show_data_overview:
        st.write("### Data Overview - sizes-combined-indoor", df)

    if "x" in df.columns and "y" in df.columns:
        size_options = ["All"] + df["sizes"].unique().tolist()
        selected_size = st.sidebar.selectbox("Select Size", size_options)

        if selected_size == "All":
            filtered_deployment_codes = df["deployment-code"].unique()
        else:
            filtered_deployment_codes = df[df["sizes"] == selected_size][
                "deployment-code"
            ].unique()

        deployment_code = st.sidebar.selectbox(
            "Select Deployment Code", filtered_deployment_codes
        )

        st.sidebar.markdown(
            """
            **Function Descriptions:**
            - **Function 1:** 1 / (a + b * x)
            - **Function 2:** 1 / (a * x + b)
            - **Function 3:** a * (1 / x) + b
            """
        )

        func_type = st.sidebar.selectbox(
            "Select Model Function", ["Function 1", "Function 2", "Function 3"]
        )

        show_filtered_data = st.sidebar.checkbox("Show Filtered Data", value=False)

        if st.sidebar.button("Display"):
            filtered_df = df[df["deployment-code"] == deployment_code]
            if selected_size != "All":
                filtered_df = filtered_df[filtered_df["sizes"] == selected_size]

            if show_filtered_data:
                st.write(f"### Data for Deployment Code {deployment_code}")
                st.write(filtered_df)

            results = {}
            for deployment_code, group in filtered_df.groupby("deployment-code"):
                a_opt, b_opt = optimize_parameters(group, func_type)
                results[deployment_code] = (a_opt, b_opt)
                st.write(f"### Deployment {deployment_code}: a = {a_opt}, b = {b_opt}")

            plot_data = []
            for deployment_code, group in filtered_df.groupby("deployment-code"):
                a_opt, b_opt = results[deployment_code]
                x_fit = np.linspace(min(group["x"]), max(group["x"]), 100)
                y_fit = model_func(x_fit, a_opt, b_opt, func_type)
                plot_data.append(
                    pd.DataFrame(
                        {"x": x_fit, "y": y_fit, "deployment-code": deployment_code}
                    )
                )

            plot_data = pd.concat(plot_data)

            points = (
                alt.Chart(filtered_df)
                .mark_circle(size=60)
                .encode(
                    x="x",
                    y="y",
                    color="deployment-code:N",
                    tooltip=["x", "y", "deployment-code"],
                )
                .properties(width=800, height=600)
            )

            lines = (
                alt.Chart(plot_data)
                .mark_line()
                .encode(
                    x="x",
                    y="y",
                    color="deployment-code:N",
                    tooltip=["x", "y", "deployment-code"],
                )
            )

            chart = points + lines
            st.altair_chart(chart, use_container_width=True)
    else:
        st.write("The required columns 'x' and 'y' are not present in the DataFrame.")

elif page == "Depth Calculation":
    st.title("Depth Calculation")

    # Load data
    df = load_data(CSV_PATH)

    st.sidebar.title("Filter Options")
    # Select size
    size_options = ["All"] + df["sizes"].unique().tolist()
    selected_size = st.sidebar.selectbox("Select Size", size_options, key="size_select")

    if selected_size != "All":
        df = df[df["sizes"] == selected_size]

    # Select deployment code
    deployment_options = ["All"] + df["deployment-code"].unique().tolist()
    selected_deployment = st.sidebar.selectbox(
        "Select Deployment Code", deployment_options, key="deployment_select"
    )

    if selected_deployment != "All":
        df = df[df["deployment-code"] == selected_deployment]

    # Inputs for zmin, zmax with validation
    zmin = st.slider("**Select zmin**", 0.0, 20.0, 0.0, key="zmin_slider")
    zmax = st.slider("**Select zmax**", 0.0, 50.0, 0.0, key="zmax_slider")

    if zmin >= zmax:
        st.warning("zmin must be less than zmax. Please adjust the sliders.")
    else:
        # Calculate value_metric for each row
        depth_value_columns = ["depth_value_indoor_left", "depth_value_outdoor_left"]
        for depth_value_column in depth_value_columns:
            if depth_value_column in df.columns:
                df[f"value_metric_{depth_value_column}"] = df.apply(
                    lambda row: calculate_value_metric(
                        row[depth_value_column], zmin, zmax
                    ),
                    axis=1,
                )

        # Function to calculate length_metric
        def calculate_length_metric(row, value_metric_col):
            x0, y0 = row["Lx0"], row["Ly0"]  # Always use L camera
            x1, y1 = row["Lx1"], row["Ly1"]
            dx = x1 - x0
            dy = y1 - y0
            length_pixels = np.sqrt(dx**2 + dy**2)
            return (row[value_metric_col] * length_pixels) / row["focal_length"]

        # Calculate predicted sizes for indoor and outdoor
        df["predicted_size_indoor"] = df.apply(
            lambda row: calculate_length_metric(
                row, "value_metric_depth_value_indoor_left"
            ),
            axis=1,
        )
        df["predicted_size_outdoor"] = df.apply(
            lambda row: calculate_length_metric(
                row, "value_metric_depth_value_outdoor_left"
            ),
            axis=1,
        )

        # Calculate errors
        df["range_error_indoor"] = (
            df["object-range"] - df["value_metric_depth_value_indoor_left"]
        )
        df["range_error_outdoor"] = (
            df["object-range"] - df["value_metric_depth_value_outdoor_left"]
        )
        df["size_error_indoor"] = df["size_mm"] - df["predicted_size_indoor"]
        df["size_error_outdoor"] = df["size_mm"] - df["predicted_size_outdoor"]

        # Show/hide options
        show_range_plot = st.sidebar.checkbox(
            "Show Plot: object-range vs predicted-range", value=True
        )
        show_size_plot = st.sidebar.checkbox(
            "Show Plot: actual size vs predicted size", value=True
        )
        show_range_hist = st.sidebar.checkbox("Show Histogram: Range Error", value=True)
        show_size_hist = st.sidebar.checkbox("Show Histogram: Size Error", value=True)

        if show_range_plot:
            # Plot value_metric vs object-range side by side
            indoor_chart = (
                alt.Chart(df)
                .mark_circle(size=60)
                .encode(
                    x="object-range",
                    y=alt.Y(
                        "value_metric_depth_value_indoor_left",
                        title="predicted-indoor-range",
                    ),
                    tooltip=["object-range", "value_metric_depth_value_indoor_left"],
                )
                .properties(width=600, height=400, title="Indoor")
            )

            outdoor_chart = (
                alt.Chart(df)
                .mark_circle(size=60)
                .encode(
                    x="object-range",
                    y=alt.Y(
                        "value_metric_depth_value_outdoor_left",
                        title="predicted-outdoor-range",
                    ),
                    tooltip=["object-range", "value_metric_depth_value_outdoor_left"],
                )
                .properties(width=600, height=400, title="Outdoor")
            )

            st.write("### Plot: object-range vs predicted-range")
            st.altair_chart(indoor_chart | outdoor_chart, use_container_width=True)

        if show_size_plot:
            # Plot size_mm vs predicted_size side by side
            size_indoor_chart = (
                alt.Chart(df)
                .mark_circle(size=60)
                .encode(
                    x="size_mm",
                    y=alt.Y("predicted_size_indoor", title="predicted-indoor size"),
                    tooltip=["size_mm", "predicted_size_indoor"],
                )
                .properties(width=600, height=400, title="Size Comparison Indoor")
            )

            size_outdoor_chart = (
                alt.Chart(df)
                .mark_circle(size=60)
                .encode(
                    x="size_mm",
                    y=alt.Y("predicted_size_outdoor", title="predicted-outdoor size"),
                    tooltip=["size_mm", "predicted_size_outdoor"],
                )
                .properties(width=600, height=400, title="Size Comparison Outdoor")
            )

            st.write("### Plot: actual size vs predicted size")
            st.altair_chart(
                size_indoor_chart | size_outdoor_chart, use_container_width=True
            )

        if show_range_hist:
            # Plot range error histograms
            range_error_indoor_hist = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X("range_error_indoor", bin=True, title="Range Error Indoor"),
                    y=alt.Y("count()", title="Frequency"),
                )
                .properties(width=600, height=400, title="Range Error Indoor Histogram")
            )

            range_error_outdoor_hist = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "range_error_outdoor", bin=True, title="Range Error Outdoor"
                    ),
                    y=alt.Y("count()", title="Frequency"),
                )
                .properties(
                    width=600, height=400, title="Range Error Outdoor Histogram"
                )
            )

            st.write("### Histogram: Range Error")
            st.altair_chart(
                range_error_indoor_hist | range_error_outdoor_hist,
                use_container_width=True,
            )

        if show_size_hist:
            # Plot size error histograms
            size_error_indoor_hist = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X("size_error_indoor", bin=True, title="Size Error Indoor"),
                    y=alt.Y("count()", title="Frequency"),
                )
                .properties(width=600, height=400, title="Size Error Indoor Histogram")
            )

            size_error_outdoor_hist = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X("size_error_outdoor", bin=True, title="Size Error Outdoor"),
                    y=alt.Y("count()", title="Frequency"),
                )
                .properties(width=600, height=400, title="Size Error Outdoor Histogram")
            )

            st.write("### Histogram: Size Error")
            st.altair_chart(
                size_error_indoor_hist | size_error_outdoor_hist,
                use_container_width=True,
            )


elif page == "Function 3 Optimized Graphs":
    st.title("Function 3 Optimized Graphs")
    df = load_data(CSV_PATH_2)

    st.sidebar.title("Graph Filter Options")
    show_data_overview = st.sidebar.checkbox("Show Data Overview", value=True)
    if show_data_overview:
        st.write("### Data Overview - sizes-combined-indoor", df)

    if "x" in df.columns and "object-range" in df.columns:
        size_options = ["All"] + df["sizes"].unique().tolist()
        selected_size = st.sidebar.selectbox("Select Size", size_options)

        if selected_size == "All":
            filtered_deployment_codes = df["deployment-code"].unique()
        else:
            filtered_deployment_codes = df[df["sizes"] == selected_size][
                "deployment-code"
            ].unique()

        deployment_code = st.sidebar.selectbox(
            "Select Deployment Code", filtered_deployment_codes
        )

        if st.sidebar.button("Display"):
            filtered_df = df[df["deployment-code"] == deployment_code]
            if selected_size != "All":
                filtered_df = filtered_df[filtered_df["sizes"] == selected_size]

            results = {}
            for deployment_code, group in filtered_df.groupby("deployment-code"):
                a_opt, b_opt = optimize_parameters(group, "Function 3")
                results[deployment_code] = (a_opt, b_opt)
                st.write(f"### Deployment {deployment_code}: a = {a_opt}, b = {b_opt}")

            # Calculate new_value_metric using the optimized parameters and Function 3
            new_value_metrics = []
            for deployment_code, group in filtered_df.groupby("deployment-code"):
                a_opt, b_opt = results[deployment_code]
                new_value_metric = model_func(group["x"], a_opt, b_opt, "Function 3")
                new_value_metrics.append(
                    pd.DataFrame(
                        {
                            "object-range": group["object-range"],
                            "new_value_metric": new_value_metric,
                            "deployment-code": deployment_code,
                        }
                    )
                )

            new_value_metrics_df = pd.concat(new_value_metrics)

            # Plot new_value_metric vs object-range
            chart = (
                alt.Chart(new_value_metrics_df)
                .mark_circle(size=60)
                .encode(
                    x="object-range",
                    y="new_value_metric",
                    color="deployment-code:N",
                    tooltip=["object-range", "new_value_metric", "deployment-code"],
                )
                .properties(width=800, height=600)
            )

            st.write("### Plot: object-range vs new_value_metric")
            st.altair_chart(chart, use_container_width=True)
    else:
        st.write(
            "The required columns 'x' and 'object-range' are not present in the DataFrame."
        )
