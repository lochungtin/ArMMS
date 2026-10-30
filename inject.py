import json
from argparse import ArgumentParser
from os.path import dirname, join

from src.comparison import compareAll, compareSeries, loadMarkers


def getArgs():
    ap = ArgumentParser()
    ap.add_argument("markers", type=str)  # path to markers.json
    ap.add_argument("-s", "--size", type=float, default=3)  # marker size in any unit
    ap.add_argument("-m", "--mode", type=str, default="all")  # comparison mode
    # output file, defaults to injected_results.json next to markers.json
    ap.add_argument("-o", "--output", type=str)
    return ap.parse_args()


if __name__ == "__main__":
    args = getArgs()
    markers = loadMarkers(args.markers)

    compare = compareSeries if args.mode == "series" else compareAll
    results = {grpName: compare(grp, args.size) for grpName, grp in markers.items()}

    out = args.output or join(dirname(args.markers), "injected_results.json")
    with open(out, "w") as w:
        json.dump(results, w)
