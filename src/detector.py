from pathlib import Path

import cv2 as cv
import cv2.aruco as arc
import numpy as np

# Marker type to string dictionary, for command parameters
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


class Detector:
    def __init__(self, markerType, advancedThresholding, logger, markerFilter=None):
        self.mType = markerType
        self.advThres = advancedThresholding
        self.mFilter = markerFilter
        self.logger = logger

        if self.advThres:
            logger.warn("Advanced thresholding is enabled.")

        # load aruco marker dictionary
        if self.mType in ARUCO_DICT:
            self.dict = arc.getPredefinedDictionary(ARUCO_DICT[self.mType])
            logger.action(f"Logger created: [{self.mType}] type selected")
        else:
            logger.error(f"ArUCo tag type {self.mType} is not a valid type.")
            exit(1)

        self.params = arc.DetectorParameters_create()

    def detectMarkersFromFile(self, filename):
        text = f"Started detection for {Path(filename).name}"
        self.logger.action(text, toConsole=self.logger.verbose, noNewLine=1)
        return self.detectMarkers(cv.imread(filename))

    def _detect(self, img):
        # get corners and ids of an image
        corners, ids, _ = arc.detectMarkers(img, self.dict, parameters=self.params)

        if len(corners) == 0:
            return [], []

        # pair up ids and corners into tuples
        pairs = [(i[0], c) for i, c in zip(ids, corners)]
        pairs.sort(key=lambda x: x[0])

        # filter out unwanted detections
        if self.mFilter:
            return list(filter(lambda x: x[0] in self.mFilter, pairs))

        return pairs

    def _output(self, s):
        ids = np.array(list(s.keys())).reshape(-1, 1)
        markers = tuple(s.values())

        text = f" -> Detected markers: {' '.join(str(x[0]) for x in ids)}"
        self.logger.action(text, toConsole=self.logger.verbose)

        return ids, markers

    def detectMarkers(self, image):
        store = {}
        # original color space
        for id, c in self._detect(image):
            store[id] = c

        if not self.advThres:
            return self._output(store)

        # advanced thresholding by varying brightness and contrast of the original image
        for a in range(10, 40, 2):
            for g in range(20, 2):
                br = cv.addWeighted(image, a / 10, image, 0, float(g))
                ad = cv.convertScaleAbs(image, alpha=a, beta=g)
                mono = cv.cvtColor(
                    cv.cvtColor(br, cv.COLOR_RGB2GRAY), cv.COLOR_GRAY2RGB
                )
                for alt in [br, ad, mono]:
                    for id, c in self._detect(alt):
                        store[id] = c

                for i in range(60, 120, 2):
                    _, thres = cv.threshold(mono, i, 255, cv.THRESH_BINARY)
                    for id, c in self._detect(thres):
                        store[id] = c

        # swap color space
        for id, c in self._detect(cv.cvtColor(image, cv.COLOR_RGB2BGR)):
            store[id] = c

        return self._output(store)

    def drawMarkersFromFile(self, filename, ids, corners):
        return self.drawMarkers(cv.imread(filename), ids, corners)

    def drawMarkers(self, image, ids, corners):
        # highlight markers
        image = arc.drawDetectedMarkers(image, corners)
        # add marker id text
        for i, c in zip(ids, corners):
            loc = (int(c[0][0][0]), int(c[0][0][1] - 15))
            cv.putText(image, str(i), loc, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        return image
