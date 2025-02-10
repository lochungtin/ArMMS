from argparse import ArgumentParser
from itertools import combinations as comb
from os import listdir, makedirs
from os.path import exists, isfile, join

import cv2 as cv
import numpy as np
from detector import Detector
from logger import Logger, _timestamp
from marker import Marker
from PIL import Image
from tqdm import tqdm
from utils import load, save


def getArguments():
    ap = ArgumentParser()
    # aruco marker type
    ap.add_argument("-t", "--type", type=str, default="DICT_4X4_100")
    # aruco marker size in any unit
    ap.add_argument("-s", "--size", type=float, default=3)
    # image list json file
    ap.add_argument("-j", "--job", type=str, default="jobs/inference.json")
    # output directory
    ap.add_argument("-o", "--output", type=str, default="out")
    # generate result images
    ap.add_argument("-g", "--generate", type=int, default=1)
    # generate overlaid images
    ap.add_argument("-x", "--overlay", type=int, default=1)
    # enable advanced thresholding
    ap.add_argument("-a", "--advanced", type=int, default=0)
    # do detection only
    ap.add_argument("-d", "--detection", type=int, default=0)

    args = vars(ap.parse_args())
    return tuple(args[x] for x in args.keys())


def jsonValidation(inJson, logger):
    # check if input json exists
    if not exists(inJson):
        logger.error("Input JSON not found.")

    data = load(inJson)

    # check if description tag is in json
    if "description" not in data:
        logger.error(
            "Key <description> not found in input JSON, please referenece sample.json."
        )

    # check if description is a list
    elif not isinstance(data["description"], list):
        logger.error(
            "Value of key <description> in input JSON not of type List, please referenece sample.json."
        )

    # check if data tag is in json
    if "data" not in data:
        logger.error(
            "Key <data> not found in input JSON, please referenece sample.json."
        )

    # validate contents of input JSON
    def traverse(layer, layerNum, tags, images):
        # check if image path data matches description
        if layerNum == len(data["description"]):
            if not isinstance(layer, list):
                logger.error(
                    "Image path data in input JSON does not match description."
                )
            # validate image paths
            for i, v in enumerate(layer):
                images[tuple(tags + [str(i + 1)])] = v
            return
        elif not isinstance(layer, dict):
            logger.error("Image path data in input JSON does not match description.")

        # check if layer has content
        if len(layer) == 0:
            logger.error("Image path data in input JSON contains an empty dictionary.")

        # recursive step
        for k, v in layer.items():
            traverse(v, layerNum + 1, tags + [k], images)

    # retrieve list of images
    images = {}
    traverse(data["data"], 0, [], images)
    return images, data


def initialisation(inJson, willOutput, willOverlay, rootDir, logger):
    # validate input json
    logger.action("Validating input JSON ...")
    images, data = jsonValidation(inJson, logger)

    # validate images
    logger.action("Validating image paths ...")
    for _, path in tqdm(images.items()):
        if not isfile(path):
            logger.error(f"Failed to read image:\n{path}")
    print()

    # create output directories
    if willOutput:
        logger.action("Creating leaf directories ...")
        for k in tqdm(sorted(list(images.keys()))):
            makedirs(join(rootDir, "images", *k[:-1]), exist_ok=True)
            if willOverlay:
                makedirs(join(rootDir, "images", *k[:-1], "overlay"), exist_ok=True)
    print()
    return images, data


def detection(mType, willOutput, advThres, images, logger):
    # load allowed markers
    validMarkers = set(data["valid_ids"]) if "valid_ids" in data else False

    # Create detector
    detector = Detector(mType, advThres, validMarkers, logger)

    logger.action("Starting detection process ...")

    results = {}
    for keys, image in tqdm(images.items()):
        # load image and detec markers
        ids, markers = detector.detectMarkersFromFile(image)

        # create data hierarchy
        topLayer = results
        for key in keys[:-1]:
            if key not in topLayer:
                topLayer[key] = {}
            topLayer = topLayer[key]

        # save vertices
        for id, marker in zip(ids, markers):
            id = int(id[0])
            if id not in topLayer:
                topLayer[id] = {}
            topLayer[id][keys[-1]] = marker[0].astype(int).tolist()

        # generate output image
        if willOutput:
            outputImg = detector.drawMarkers(cv.imread(image), ids, markers)
            outputRes = cv.imwrite(
                join(rootDir, "images", *keys[:-1], "_".join(keys) + ".png"), outputImg
            )
            logOut = f" -> Output file {'saved successfully' if outputRes else 'failed to save'}."
            logger.action(logOut, toConsole=0)
    print()
    return results


def overlay(images, rootDir, logger):
    logger.action("Computing overlays ...")
    for grp in tqdm(list(set(keys[:-1] for keys in images.keys()))):
        # group files
        files = []
        directory = join(rootDir, "images", *grp)
        for fname in listdir(directory):
            file = join(directory, fname)
            if isfile(file):
                files.append(Image.open(file).convert("RGBA"))

        # create base file
        size = files[0].size
        output = Image.new("RGBA", size)
        alpha = int(255 / len(files))

        # alpha composite overlays
        for img in files:
            img.putalpha(Image.new("L", size, alpha))
            output = Image.alpha_composite(output, img)

        # save output
        output.convert("RGB").save(join(directory, "overlay", "result.png"))
    print()


def comparison(mSize, images, results, logger):
    # get groupings, ignore ids
    groupings = set()
    for keys in images.keys():
        groupings.add(keys[:-1])

    logger.action("Calculating variations ...")
    for keys in tqdm(sorted(list(groupings))):
        # get top layer, just before repeats
        topLayer = results
        for key in keys:
            topLayer = topLayer[key]

        for data in topLayer.values():
            og_keys = list(data.keys())
            diffs = []
            # loop over repeat number combinations
            for k0, k1 in comb(sorted(og_keys), 2):
                m0, m1 = Marker(data[k0]), Marker(data[k1])
                _, d = m0 - m1
                l = (m0.recoverEdgeLength() + m1.recoverEdgeLength()) / 2
                data[f"{k0}-{k1}"] = mSize / l * d
                diffs.append(data[f"{k0}-{k1}"])
            # calculate average between repeats
            if len(diffs):
                data["avg"] = np.mean(diffs, axis=0).tolist()
            for key in og_keys:
                del data[key]
    print()
    return results


if __name__ == "__main__":
    mType, mSize, inJson, outDir, willOutput, willOverlay, advThres, onlyDetect = (
        getArguments()
    )

    # Create logger
    rootDir = join(outDir, f"JOB_{_timestamp(True)}")
    makedirs(rootDir)
    logger = Logger(rootDir)

    # initialise directories
    logger.plain("======== JOB INITIALISATION ========\n")
    images, data = initialisation(inJson, willOutput, willOverlay, rootDir, logger)

    # detect markers from images
    logger.plain("======== DETECTION STEP ========\n")
    results = detection(mType, willOutput, advThres, images, logger)

    # write all vertices to file
    logger.action("Detection complete, writing data to file ...\n")
    save(results, join(rootDir, "markers.json"))

    # generate overlays
    if willOutput and willOverlay:
        logger.plain("======== GENERATE OVERLAYS ========\n")
        overlay(images, rootDir, logger)

    if onlyDetect:
        exit(0)

    # evaluate marker position variation
    logger.plain("======== COMPARISON STEP ========\n")
    results = comparison(mSize, images, results, logger)

    # write all distance results to file
    logger.action("Calculation complete, writing data to file ...\n")
    save(results, join(rootDir, "results.json"))

    logger.close()
