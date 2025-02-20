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


def main():
    ap = ArgumentParser()
    ap.add_argument("-t", "--type", type=str, default="DICT_4X4_100")  # aruco marker type
    ap.add_argument("-s", "--size", type=float, default=3)  # aruco marker size in any unit
    ap.add_argument("-i", "--ids", nargs="+", default=[])  # valid arcuo marker ids
    ap.add_argument("-r", "--root", type=str, default="data/")  # image list json file
    ap.add_argument("-o", "--output", type=str, default="out")  # output directory
    ap.add_argument("-g", "--generate", type=int, default=1)  # generate result images
    ap.add_argument("-x", "--overlay", type=int, default=1)  # generate overlaid images
    ap.add_argument("-a", "--advanced", type=int, default=0)  # enable advanced thresholding
    ap.add_argument("-d", "--detection", type=int, default=0)  # do detection only
    ap.add_argument("-v", "--verbose", type=int, default=1)  # do command line output
    args = vars(ap.parse_args())
    (
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
    ) = tuple(args[x] for x in args.keys())

    VALID_MARKER_IDS = list(map(int, VALID_MARKER_IDS))

    OUT_DIR = join(OUT_DIR, f"JOB_{_timestamp(True)}")
    makedirs(OUT_DIR)

    logger = Logger(OUT_DIR)
    detector = Detector(DICT_TYPE, USE_ADV_THRESH, VALID_MARKER_IDS, logger)

    logger.plain("=== INITIALISATION: Verifying Subdirectories ===")
    markers, _markers, imgs = {}, {}, []
    subDirs = natsorted(listdir(ROOT_DIR))
    for subDirName in subDirs if VERBOSE else tqdm(subDirs):
        if subDirName != ".DS_Store":
            logger.action(f"Checking folder: {subDirName}", toConsole=VERBOSE)
            folder = join(ROOT_DIR, subDirName)
            fList = natsorted(listdir(folder))

            if len(fList) < 2:
                logger.warn(f"Skipping directory - directory has less than 2 images")

            markers[subDirName] = dict(zip(fList, [{}] * len(fList)))
            _markers[subDirName] = dict(zip(fList, [{}] * len(fList)))
            for fileName in fList:
                if fileName != ".DS_Store":
                    imgs.append((subDirName, fileName))

            if GEN_RESULTS:
                makedirs(join(OUT_DIR, "images", subDirName))
                if GEN_OVERLAYS:
                    makedirs(join(OUT_DIR, "images", subDirName, "overlay"))

    logger.plain("=== DETECTION ===")
    for subDirName, fileName in imgs if VERBOSE else tqdm(imgs):
        img = join(ROOT_DIR, subDirName, fileName)
        i, m = detector.detectMarkersFromFile(img, VERBOSE)

        if GEN_RESULTS:
            res = detector.drawMarkersFromFile(img, i, m)
            sOut = cv.imwrite(join(OUT_DIR, "images", subDirName, fileName), res)
            logOut = f" -> Output file {'saved successfully' if sOut else 'failed to save'}."
            logger.action(logOut, toConsole=VERBOSE)

        i = list(map(str, i.flatten()))
        m = [x[0].tolist() for x in m]
        markers[subDirName][fileName] = dict(zip(i, m))
        _markers[subDirName][fileName] = dict(zip(i, map(Marker, m)))

    saveJSON(markers, join(OUT_DIR, "markers.json"))

    rows = []
    for grpName, grp in markers.items():
        for imgName, img in grp.items():
            for mID, m in img.items():
                rows.append(
                    [grpName, imgName, mID, *list(map(str, np.array(m).flatten().tolist()))]
                )
    saveCSV(MARKER_CSV_HEAD, rows, join(OUT_DIR, "markers.csv"))

    if not DETECTION_ONLY:
        logger.plain("=== CALCULATION AND COMPARISON ===")
        results = {}
        for grpName, grp in _markers.items() if VERBOSE else tqdm(_markers.items()):
            results[grpName] = {}
            logger.action(f"Processing group: {grpName}", toConsole=VERBOSE, noNewLine=1)
            for i0, i1 in comb(grp.keys(), 2):
                pair = f"{i0}/{i1}"
                s0, s1 = set(grp[i0].keys()), set(grp[i1].keys())
                ms = sorted(list(s0.intersection(s1)))
                logger.action(f"Matching pairs found: {' '.join(ms)}", toConsole=VERBOSE)
                for mID in ms:
                    m0, m1 = grp[i0][mID], grp[i1][mID]
                    [x, y], d = m0 - m1
                    r = MARKER_SIZE / ((m0.recoverEdgeLength() + m1.recoverEdgeLength()) / 2)
                    if mID not in results[grpName]:
                        if mID not in results[grpName]:
                            results[grpName][mID] = {}
                        results[grpName][mID][pair] = {"dx": r * x, "dy": r * y, "d": r * d}

        saveJSON(results, join(OUT_DIR, "results.json"))
        rows = []
        for grpName, grp in results.items():
            for mID, pairs in grp.items():
                for imgName, img in pairs.items():
                    img0, img1 = imgName.split("/")
                    rows.append(
                        [grpName, mID, img0, img1, str(img["dx"]), str(img["dy"]), str(img["d"])]
                    )
        saveCSV(RESULTS_CSV_HEAD, rows, join(OUT_DIR, "results.csv"))

    if GEN_RESULTS and GEN_OVERLAYS:
        logger.plain("=== COMPUTING OVERLAYS ===")
        for grpName, grp in markers.items() if VERBOSE else tqdm(markers.items()):
            logger.action(f"Processing group: {grpName}", toConsole=VERBOSE, noNewLine=1)
            folder = join(OUT_DIR, "images", grpName)
            imgFs = [Image.open(join(folder, i)).convert("RGBA") for i in natsorted(grp.keys())]

            if len(imgFs) == 1:
                logger.action(
                    " -> Directory only has 1 image, no overlays will be generated",
                    toConsole=VERBOSE,
                )
            else:
                output = Image.new("RGBA", imgFs[0].size)
                alpha = int(255 / len(imgFs))

                for img in imgFs:
                    img.putalpha(Image.new("L", imgFs[0].size, alpha))
                    output = Image.alpha_composite(output, img)

                output.convert("RGB").save(join(folder, "overlay", "result.png"))
                logger.action(" -> Overlay generated", toConsole=VERBOSE)

    logger.close()


if __name__ == "__main__":
    main()
