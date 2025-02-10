from argparse import ArgumentParser
from os import mkdir
from os.path import exists, isdir, join
from pathlib import Path

import numpy as np
from logger import Logger
from utils import load, save


def getArguments():
    ap = ArgumentParser()
    # analysis file
    ap.add_argument("-j", "--job", type=str, default=Path("jobs/analysis.json"))
    # output directory
    ap.add_argument("-o", "--output", type=str, default="out/")

    args = vars(ap.parse_args())
    return tuple(args[x] for x in args.keys())


def jsonValidation(conf, res, logger):
    if "description" not in conf or "groupBy" not in conf:
        logger.error("Analysis JSON must contain tags <description> and <groupBy>.")

    def traverse(layer, layerNum, tags, data):
        if not isinstance(layer, dict):
            logger.error("Results JSON may be corrupted, exiting.")
        if "avg" in layer:
            data[tuple(tags)] = layer["avg"]
            return layerNum
        if len(layer) == 0:
            return layerNum
        return max(
            traverse(v, layerNum + 1, tags + [k], data) for k, v in layer.items()
        )

    data = {}
    if len(conf["description"]) != traverse(res, 0, [], data):
        logger.error(
            "<description> in analysisJ SON does not match the structure of results.json"
        )

    for grp in conf["groupBy"]:
        for item in grp:
            if item not in conf["description"]:
                logger.error(f"Group by item [{item}] not part of description.")
    return data


def stats(arr):
    return float(np.mean(arr))


if __name__ == "__main__":
    inJson, outDir = getArguments()

    # create logger
    if not isdir(outDir):
        print("Error: out directory does not exist")
        exit(1)
    logger = Logger(outDir)

    # region - [Job Initialisation]
    logger.plain("======== JOB INITIALISATION ========\n")

    # load analysis config
    logger.action("Loading analysis JSON ...")
    analysisConf = {}
    if exists(inJson):
        analysisConf = load(inJson)
    else:
        logger.error("Analysis JSON not found.")

    # load result json
    logger.action("Loading results.json ...")
    resJson = join(outDir, "results.json")
    results = {}
    if exists(resJson):
        results = load(resJson)
    else:
        logger.error(f"result.json was not found in {outDir}.")

    # validate input jsons
    logger.action("Validating input JSONs and configurations ...")
    data = jsonValidation(analysisConf, results, logger)

    _data = dict(zip(map(lambda x: ":".join(x), data.keys()), data.values()))
    save(_data, join(outDir, "result_tupled.json"))

    # create output directory
    logger.action("Creating output directory ...")
    outDir = join(outDir, "analysis")
    if not exists(outDir):
        mkdir(outDir)
    else:
        logger.warn("Analysis directory already exists, files will be overwritten.")
    # endregion

    for job in analysisConf["groupBy"]:
        jobName = "_".join(job)
        logger.action(f"Starting analysis for [{jobName}]")
        out = {}

        # group data entries
        idx = [analysisConf["description"].index(target) for target in job]
        for key, value in data.items():
            newKey = "grp:" + "_".join(k for i, k in enumerate(key) if i not in idx)
            if newKey not in out:
                out[newKey] = []
            out[newKey].append(value)

        # calculate stats
        for key, values in out.items():
            out[key] = np.mean(values).astype(float)

        # write data to file
        print(outDir)
        save(out, join(outDir, f"grp_{jobName}.json"))
