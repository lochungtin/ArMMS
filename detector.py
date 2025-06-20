from pathlib import Path

import cv2 as cv
import numpy as np
from cv2 import aruco as arc
from utils import loadArUcoDict


class Detector:
    def __init__(self, markerType, advancedThresholding, markerFilter=None, logger=None):
        self.mType = markerType
        self.advThres = advancedThresholding
        self.mFilter = markerFilter
        self.logger = logger

        if self.advThres and self.logger:
            logger.warn("Advanced thresholding is enabled.")

        # load aruco marker dictionary
        self.dict = loadArUcoDict(self.mType)
        if self.dict == None:
            if self.logger:
                logger.error(f"ArUCo tag type {self.mType} is not a valid type.")
            else:
                print(f"Error: Marker type {self.mType} is not valid.")
            exit(1)
        else:
            if self.logger:
                logger.action(f"Logger created: [{self.mType}] type selected")
        self.params = arc.DetectorParameters_create()

    def detectMarkersFromFile(self, filename, verbose=False):
        if self.logger:
            text = f"Started detection for {Path(filename).name}"
            self.logger.action(text, toConsole=verbose, noNewLine=1)
        return self.detectMarkers(cv.imread(filename), verbose)

    def detectMarkers(self, image, verbose=False):
        def detect(img):
            # get corners and ids
            corners, ids, _ = arc.detectMarkers(img, self.dict, parameters=self.params)

            if len(corners) == 0:
                return [], []

            # filter out unwanted detections
            pairs = [(i[0], c) for i, c in zip(ids, corners)]
            if self.mFilter:
                pairs.sort(key=lambda x: x[0])
                return zip(*list(filter(lambda x: x[0] in self.mFilter, pairs)))

            return zip(*pairs)

        def output(s):
            # reshape output
            ids = np.array(list(s.keys())).reshape(-1, 1)
            markers = tuple(s.values())

            if self.logger:
                mStr = " ".join(list(map(lambda x: str(x[0]), ids)))
                self.logger.action(f" -> Detected markers: {mStr}", toConsole=verbose)

            return ids, markers

        store = {}
        # original color space
        for id, c in zip(*detect(image)):
            store[id] = c

        if not self.advThres:
            # if len(store) == 8:  # for validation only, uncomment line above to use
            return output(store)

        # advanced thresholding with varying brightness and contrast
        for a in range(10, 40, 2):
            for g in range(20, 2):
                br = cv.addWeighted(image, a / 10, image, 0, float(g))
                ad = cv.convertScaleAbs(image, alpha=a, beta=g)
                mono = cv.cvtColor(cv.cvtColor(br, cv.COLOR_RGB2GRAY), cv.COLOR_GRAY2RGB)
                for alt in [br, ad, mono]:
                    for id, c in zip(*detect(alt)):
                        store[id] = c

                for i in range(60, 120, 2):
                    _, thres = cv.threshold(mono, i, 255, cv.THRESH_BINARY)
                    for id, c in zip(*detect(thres)):
                        store[id] = c

        # swap color space
        bgr = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        for id, c in zip(*detect(bgr)):
            store[id] = c

        return output(store)

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
