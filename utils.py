import json

import cv2.aruco as arc
import numpy as np

ARUCO_DICT = {
    "DICT_4X4_50": arc.DICT_4X4_50,
    "DICT_4X4_100": arc.DICT_4X4_100,
    "DICT_4X4_250": arc.DICT_4X4_250,
    "DICT_4X4_1000": arc.DICT_4X4_1000,
    "DICT_5X5_50": arc.DICT_5X5_50,
    "DICT_5X5_100": arc.DICT_5X5_100,
    "DICT_5X5_250": arc.DICT_5X5_250,
    "DICT_5X5_1000": arc.DICT_5X5_1000,
    "DICT_6X6_50": arc.DICT_6X6_50,
    "DICT_6X6_100": arc.DICT_6X6_100,
    "DICT_6X6_250": arc.DICT_6X6_250,
    "DICT_6X6_1000": arc.DICT_6X6_1000,
    "DICT_7X7_50": arc.DICT_7X7_50,
    "DICT_7X7_100": arc.DICT_7X7_100,
    "DICT_7X7_250": arc.DICT_7X7_250,
    "DICT_7X7_1000": arc.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": arc.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": arc.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": arc.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": arc.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": arc.DICT_APRILTAG_36h11,
}

CORNER2EDGE_MAP = [[0, 1], [1, 2], [2, 3], [3, 0]]


def loadAruCoDict(mType):
    """Load OpenCV AruCo Marker Dictionary

    Args:
        mType (str): Name of marker type

    Returns:
        Dict: Marker dictionar
    """
    if mType in ARUCO_DICT:
        return arc.getPredefinedDictionary(ARUCO_DICT[mType])
    return None


def load(fn):
    """Load JSON file data

    Args:
        fn (str): Filename

    Returns:
        dict[any, any]: Data dictionary
    """
    with open(fn, "r") as r:
        return json.load(r)


def save(obj, fn):
    """Save dictionary as JSON file

    Args:
        obj (dict[any, any]): Data dictionary
        fn (filename): Filename
    """
    with open(fn, "w") as w:
        json.dump(obj, w)


def dsc(i0, i1):
    """Calculate Dice Score between 2 images

    Args:
        i0 (NDArray): Image 0
        i1 (NDArray): Image 1

    Returns:
        float: Dice score
    """
    i0 = (i0 == 255).astype(int)
    i1 = (i1 == 255).astype(int)
    return np.sum(i1[i0 == 1]) * 2.0 / (np.sum(i0) + np.sum(i1))


def angDist(R0, R1):
    """Calculate angular distance between 2 rotation matrices

    Args:
        R0 (NDArray): Rotation matrix 1
        R1 (NDArray): Rotation matrix 2

    Returns:
        float: Angular Ddistance in degrees
    """
    relR = np.dot(R0.T, R1)
    traceR = np.trace(relR)
    ang = np.clip((traceR - 1) / 2, -1.0, 1.0)
    return np.degrees(np.arccos(ang))


def toHex(n):
    """Converts decimal 0 - 15 to hex 0 - f

    Args:
        n (int): Number

    Returns:
        str: Hex string
    """
    return str(hex(n))[-1]
