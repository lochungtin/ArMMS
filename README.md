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

### Inference

Run the `run.sh` to run the inference script with the provided test data. The output directory of the test should be in the `out` folder.

#### Flags

<<<<<<< Updated upstream
-   `-t` aruco marker type
-   `-s` aruco marker size in any unit
-   `-j` image list json file
-   `-o` output directory
-   `-g` generate result images
-   `-x` generate overlaid images
-   `-a` enable advanced thresholding
-   `-d` do detection only
=======
-   `DICT_TYPE`: ArUCo marker dictionary
-   `MARKER_SIZE`: Marker size (can be in any unit)
-   `VALID_MARKER_IDS`: Detection will only match the IDs provided below, leave the list empty if you want possible markers to be detected, highly advised to be used if `USE_ADV_THRESH` is enabled
-   `ROOT_DIR`: Root directory of the images, see below for file structure
-   `OUT_DIR`: Output directory for the results, log file, and generated images, **however, the output directory can't be inside the `ROOT_DIR`**
-   `GEN_RESULTS`: Boolean flag for generating result images
-   `GEN_OVERLAYS`: Boolean flag for generating an overlaid result of all the result images of the subdirectory
-   `USE_ADV_THRESH`: Boolean flag for enabling advanced thresholding. Sweeps for a multitude of thresholding settings, aids in detecting markers that are less visible.
-   `DETECTION_ONLY`: Boolean flag for disabling comparison and calculations
-   `VERBOSE`: Boolean flag for enabling command line output
>>>>>>> Stashed changes

### Analysis

To run the analysis script `analysis.sh`, first edit the analysis by changing the `JOB_ID` in the script to the name of the output folder of the inference script, it should be of the format "JOB_YYYY_MM_DD_HH_mm_ss".
