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


def loadArUcoDict(mType):
    if mType in ARUCO_DICT:
        return arc.getPredefinedDictionary(ARUCO_DICT[mType])
    return None


def load(fn):
    with open(fn, "r") as r:
        return json.load(r)


def saveJSON(obj, fn):
    with open(fn, "w") as w:
        json.dump(obj, w)


def saveCSV(header, rows, fn):
    with open(fn, "w") as w:
        for r in [header] + rows:
            w.write(f"{', '.join(r)}\n")
