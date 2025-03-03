from argparse import ArgumentParser
from itertools import combinations as comb
from os import listdir, makedirs
from os.path import join

import cv2 as cv
import numpy as np
from detector import Detector
from logger import Logger, _timestamp
from marker import Marker
from natsort import natsorted
from PIL import Image
from tqdm import tqdm
from utils import saveCSV, saveJSON

MARKER_CSV_HEAD = [
    "group",
    "image",
    "marker_id",
    "v0_x",
    "v0_y",
    "v1_x",
    "v1_y",
    "v2_x",
    "v2_y",
    "v3_x",
    "v3_y",
]
RESULTS_CSV_HEAD = ["group", "marker_id", "image0", "image1", "dx", "dy", "d"]


def getArgs():
    ap = ArgumentParser()
    ap.add_argument("-t", "--type", type=str, default="DICT_4X4_100")  # aruco marker type
    ap.add_argument("-s", "--size", type=float, default=3)  # aruco marker size in any unit
    ap.add_argument("-i", "--ids", nargs="+", default=[])  # valid arcuo marker ids
    ap.add_argument("-r", "--root", type=str, default="data/")  # image list json file
    ap.add_argument("-o", "--output", type=str, default="out/")  # output directory
    ap.add_argument("-g", "--generate", type=int, default=1)  # generate result images
    ap.add_argument("-x", "--overlay", type=int, default=1)  # generate overlaid images
    ap.add_argument("-a", "--advanced", type=int, default=0)  # enable advanced thresholding
    ap.add_argument("-d", "--detection", type=int, default=0)  # do detection only
    ap.add_argument("-v", "--verbose", type=int, default=1)  # do command line output
    args = vars(ap.parse_args())
    return tuple(args[x] for x in args.keys())


class App:
    def __init__(
        self,
        DICT_TYPE,
        MARKER_SIZE,
        VALID_MARKER_IDS,
        ROOT_DIR,
        OUT_DIR,
        GEN_RESULTS,
        GEN_OVERLAYS,
        USE_ADV_THRESH,
        DETECTION_ONLY,
        VERBOSE,
    ):
        self.DICT_TYPE = DICT_TYPE
        self.MARKER_SIZE = MARKER_SIZE
        self.VALID_MARKER_IDS = list(map(int, VALID_MARKER_IDS))
        self.ROOT_DIR = ROOT_DIR
        self.OUT_DIR = join(OUT_DIR, f"JOB_{_timestamp(True)}")
        self.GEN_RESULTS = GEN_RESULTS
        self.GEN_OVERLAYS = GEN_OVERLAYS
        self.USE_ADV_THRESH = USE_ADV_THRESH
        self.DETECTION_ONLY = DETECTION_ONLY
        self.VERBOSE = VERBOSE

        makedirs(self.OUT_DIR)

        self.logger = Logger(self.OUT_DIR)
        self.detector = Detector(
            self.DICT_TYPE, self.USE_ADV_THRESH, self.VALID_MARKER_IDS, self.logger
        )
        self.imgs = []
        self._markers = {}
        self.markers = {}
        self._results = {}
        self.results = {}

    def prepDir(self):
        self.logger.plain("=== INITIALISATION: Verifying Subdirectories ===")
        subDirs = natsorted(listdir(self.ROOT_DIR))
        for subDirName in subDirs if self.VERBOSE else tqdm(subDirs):
            if not subDirName.startswith("."):
                self.logger.action(f"Checking folder: {subDirName}", toConsole=self.VERBOSE)
                folder = join(self.ROOT_DIR, subDirName)
                fList = natsorted(listdir(folder))

                if len(fList) < 2:
                    self.logger.warn(f"Skipping directory - directory has less than 2 images")

                self.markers[subDirName] = dict(zip(fList, [{}] * len(fList)))
                self._markers[subDirName] = dict(zip(fList, [{}] * len(fList)))
                for fileName in fList:
                    if not fileName.startswith("."):
                        self.imgs.append((subDirName, fileName))

                if self.GEN_RESULTS:
                    makedirs(join(self.OUT_DIR, "images", subDirName))
                    if self.GEN_OVERLAYS:
                        makedirs(join(self.OUT_DIR, "images", subDirName, "overlay"))

    def makeIter(self, l):
        return l if self.VERBOSE else tqdm(l)

    def detection(self):
        self.logger.plain("=== DETECTION ===")
        for subDirName, fileName in self.makeIter(self.imgs):
            img = join(self.ROOT_DIR, subDirName, fileName)
            i, m = self.detector.detectMarkersFromFile(img, self.VERBOSE)

            if self.GEN_RESULTS:
                res = self.detector.drawMarkersFromFile(img, i, m)
                sOut = cv.imwrite(join(self.OUT_DIR, "images", subDirName, fileName), res)
                logOut = f" -> Output file {'saved successfully' if sOut else 'failed to save'}."
                self.logger.action(logOut, toConsole=self.VERBOSE)

                if self.GEN_OVERLAYS:
                    self._results[(subDirName, fileName)] = (i, m)

            i = list(map(str, i.flatten()))
            m = [x[0].tolist() for x in m]
            self.markers[subDirName][fileName] = dict(zip(i, m))
            self._markers[subDirName][fileName] = dict(zip(i, map(Marker, m)))

        saveJSON(self.markers, join(self.OUT_DIR, "markers.json"))
        rows = []
        for grpName, grp in self.markers.items():
            for imgName, img in grp.items():
                for mID, m in img.items():
                    rows.append(
                        [grpName, imgName, mID, *list(map(str, np.array(m).flatten().tolist()))]
                    )
        saveCSV(MARKER_CSV_HEAD, rows, join(self.OUT_DIR, "markers.csv"))

    def comparison(self):
        self.logger.plain("=== CALCULATION AND COMPARISON ===")
        for grpName, grp in self.makeIter(self._markers.items()):
            self.results[grpName] = {}
            self.logger.action(f"Processing group: {grpName}", toConsole=self.VERBOSE, noNewLine=1)
            for i0, i1 in comb(grp.keys(), 2):
                pair = f"{i0}/{i1}"
                s0, s1 = set(grp[i0].keys()), set(grp[i1].keys())
                ms = sorted(list(s0.intersection(s1)))
                self.logger.action(f"Matching pairs found: {' '.join(ms)}", toConsole=self.VERBOSE)
                for mID in ms:
                    m0, m1 = grp[i0][mID], grp[i1][mID]
                    [x, y], d = m0 - m1
                    r = self.MARKER_SIZE / ((m0.recoverEdgeLength() + m1.recoverEdgeLength()) / 2)
                    if mID not in self.results[grpName]:
                        if mID not in self.results[grpName]:
                            self.results[grpName][mID] = {}
                        self.results[grpName][mID][pair] = {"dx": r * x, "dy": r * y, "d": r * d}

        saveJSON(self.results, join(self.OUT_DIR, "results.json"))
        rows = []
        for grpName, grp in self.results.items():
            for mID, pairs in grp.items():
                for imgName, img in pairs.items():
                    img0, img1 = imgName.split("/")
                    rows.append(
                        [grpName, mID, img0, img1, str(img["dx"]), str(img["dy"]), str(img["d"])]
                    )
        saveCSV(RESULTS_CSV_HEAD, rows, join(self.OUT_DIR, "results.csv"))

    def overlays(self):
        self.logger.plain("=== COMPUTING OVERLAYS ===")
        for grpName, grp in self.makeIter(self.markers.items()):
            self.logger.action(f"Processing group: {grpName}", toConsole=self.VERBOSE, noNewLine=1)
            folder = join(self.OUT_DIR, "images", grpName)
            og = join(self.ROOT_DIR, grpName)
            imgFs = [
                Image.open(join(folder, i)).convert("RGBA")
                for i in natsorted(grp.keys())
                if not i.startswith(".")
            ]
            edges = [
                cv.Canny(
                    cv.GaussianBlur(
                        cv.cvtColor(cv.imread(join(og, i)), cv.COLOR_RGB2GRAY), (3, 3), 0
                    ),
                    threshold1=0,
                    threshold2=255,
                )
                for i in natsorted(grp.keys())
                if not i.startswith(".")
            ]

            if len(imgFs) == 1:
                self.logger.action(
                    " -> Directory only has 1 image, no overlays will be generated",
                    toConsole=self.VERBOSE,
                )
            else:
                output = Image.new("RGBA", imgFs[0].size)
                edgeOut = np.zeros((*edges[0].shape, 3))
                alpha = int(255 / len(imgFs))

                for img in imgFs:
                    img.putalpha(Image.new("L", imgFs[0].size, alpha))
                    output = Image.alpha_composite(output, img)

                for i, img in enumerate(edges):
                    color = np.zeros_like(edgeOut)
                    c = i / len(edges)
                    color[img == 255] = np.array([c, 0, 1 - c]) * 255
                    edgeOut += np.array(color)
                edgeOut = np.clip(edgeOut, 0, 255).astype(np.uint8)

                output.convert("RGB").save(join(folder, "overlay", "result.png"))
                Image.fromarray(edgeOut).save(join(folder, "overlay", "edges.png"))
                self.logger.action(" -> Overlay generated", toConsole=self.VERBOSE)

    def run(self):
        self.prepDir()
        self.detection()
        if not self.DETECTION_ONLY:
            self.comparison()
        if self.GEN_RESULTS and self.GEN_OVERLAYS:
            self.overlays()
        self.logger.close()


if __name__ == "__main__":
    app = App(*getArgs())
    app.run()
