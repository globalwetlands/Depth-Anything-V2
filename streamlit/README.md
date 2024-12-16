# Fish Data Explorer

## Overview

The Fish Data Explorer is a Streamlit application designed to facilitate the exploration and visualization of fish data from CSV files. It provides functionalities to view raw data, filter data based on various criteria, display related images, and plot graphs. This application supports optimizing parameters for predefined model functions and visualizing the results.

## Features

- **Data Overview**: View and filter the raw CSV data.
- **Image Viewer**: Display images related to the data based on user inputs.
- **Plot Graphs**: Visualize data with various types of graphs (scatter, bar, boxplot, and pie charts).
- **Optimized Graphs**: Optimize parameters for model functions and plot the results.

## Requirements

- Python 3.8+
- Streamlit
- pandas
- altair
- numpy
- scipy
- matplotlib
- Pillow

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. **Create and activate a virtual environment:**
    ```bash
    python3 -m venv streamlitenv
    source streamlitenv/bin/activate
    ```
3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
## Running the Application

1. **Activate the virtual environment:**
    ```bash
    source streamlitenv/bin/activate
    ```

2. **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

## Access the application:

- Local URL: http://localhost:8501
- Network URL: http://172.21.51.78:8501

## Application Layout and Usage

There are different script which will run different apps. Each has it's own script and dataset (images and csv).
Please read the docstring on top of each script to understand what it does.

## Example app.py
### Sidebar
The sidebar contains navigation options and filters for the different pages.

### Pages
1. **Data Overview**
- Description: View and filter the raw CSV data.
- Usage:
    - Select the "Data Overview" page.
    - Use the "Data Filter Options" in the sidebar to choose a column and value to filter the data.
    - The filtered data will be displayed in the main area.
2. **Image Viewer**
- Description: Display images related to the data based on user inputs.
- Usage:
    - Select the "Image Viewer" page.
    - Use the "Image Filter Options" in the sidebar to choose size, input camera, deployment code, and frame number.
    - Click "Display" to show the images and corresponding data.
3. **Plot Graphs**
- Description: Visualize data with various types of graphs.
- Usage:
    - Select the "Plot Graphs" page.
    - Use the "Graph Filter Options" in the sidebar to select the X-axis column, Y-axis column, and graph type.
    - Click "Display" to show the selected graph.
4. **Optimized Graphs**
- Description: Optimize parameters for model functions and plot the results.
- Usage:
    - Select the "Optimized Graphs" page.
    - Use the "Graph Filter Options" in the sidebar to select size, deployment code, and model function type.
    - Click "Display" to show the optimized graphs and parameters.