# ArUCo Marker Measurement System

## Installation

1. Create a python 3.11 virtual environment

Using Conda

```bash

conda create -n env python=3.11.5
```

Using Python VENV

```bash
python -m venv env
```

2. Install dependencies via the `requirement.txt`

```bash
pip install -r requirement.txt
```

## Basic Usage

### File Structure

The images should be structured in the following directory format.
Each subdirectory must have two or more images.

```
/path/to/root/folder
    |--> /sub_directory_1
        |--> repeat1.jpg
        |--> repeat2.jpg
    |--> /sub_directory_2
        |--> repeat1.jpg
        |--> repeat2.jpg
    |--> /sub_directory_3
        |--> repeat1.jpg
        |--> repeat2.jpg
        |--> repeat3.jpg
    ...
```

### Inference

Run the `run.sh` to run the inference script with the provided test data. The output directory of the test should be in the `out` folder.

**OR**

Open the `ipynb` notebook and run each cell, the inference process is identical.

#### Flags

-   `DICT_TYPE`: ArUCo marker dictionary
-   `MARKER_SIZE`: Marker size (can be in any unit)
-   `VALID_MARKER_IDS`: Detection will only match the IDs provided below, leave the list empty if you want possible markers to be detected, highly advised to be used if `USE_ADV_THRESH` is enabled
-   `ROOT_DIR`: Root directory of the images, see below for file structure
-   `OUT_DIR`: Output directory for the results, log file, and generated images
-   `GEN_RESULTS`: Boolean flag for generating result images
-   `GEN_OVERLAYS`: Boolean flag for generating an overlaid result of all the result images of the subdirectory
-   `USE_ADV_THRESH`: Boolean flag for enabling advanced thresholding. Sweeps for a multitude of thresholding settings, aids in detecting markers that are less visible.
-   `DETECTION_ONLY`: Boolean flag for disabling comparison and calculations
-   `VERBOSE`: Boolean flag for enabling command line output

### Inference Script Process Breakdown

**Initialisation Step**

-   Verifying that each subdirectory of the root has two or more images
-   Create leaf directories in the output folder

**Detection Step**

-   Go through each subdirectory and each image in the subdirectories
-   Detect all markers present in the images
-   Generate results images
-   Saves the intermediate results (coordinates of detected markers) to JSON and CSV file in the output directory

**Calculation and Comparison Step**

-   Recover edge length of each maker
-   Calculate the distances of markers in the same group with the same ID
-   Save result dictionary as JSON
-   Convert dictionary to rows of data and save as CSV

**Image Overlays Generation Step**

-   Go through each subdirectory in the output folder and overlay images with equal alpha
