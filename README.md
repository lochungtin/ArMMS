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

-   `-t` aruco marker type
-   `-s` aruco marker size in any unit
-   `-j` image list json file
-   `-o` output directory
-   `-g` generate result images
-   `-x` generate overlaid images
-   `-a` enable advanced thresholding
-   `-d` do detection only

### Analysis

To run the analysis script `analysis.sh`, first edit the analysis by changing the `JOB_ID` in the script to the name of the output folder of the inference script, it should be of the format "JOB_YYYY_MM_DD_HH_mm_ss".
