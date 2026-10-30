import json
from argparse import ArgumentParser
from itertools import combinations as comb
from os import makedirs
from os.path import isfile, join

from natsort import natsorted

from src.comparison import displacement, loadMarkers

LOOPS = ["loop_1", "loop_2", "loop_3"]
CAMS = ["cam_1", "cam_2"]


# compares every frame of one loop against every frame of another loop,
# for each pair of loops, grouped as {"loop_a/loop_b": {marker_id: {"img0/img1": ...}}}
def comparison_loops(markers, MARKER_SIZE, nFrames=21):
    frames = {l: natsorted(markers[l].keys())[:nFrames] for l in LOOPS}

    results = {}
    for l0, l1 in comb(LOOPS, 2):
        pair = results[f"{l0}/{l1}"] = {}
        for i0 in frames[l0]:
            for i1 in frames[l1]:
                m0s, m1s = markers[l0][i0], markers[l1][i1]
                s0, s1 = set(m0s.keys()), set(m1s.keys())

                for mID in sorted(list(s0.intersection(s1))):
                    d, dx, dy = displacement(m0s[mID], m1s[mID], MARKER_SIZE)

                    if mID not in pair:
                        pair[mID] = {}

                    pair[mID][f"{i0}/{i1}"] = {"d": d, "dx": dx, "dy": dy}

    return results, {l: len(f) for l, f in frames.items()}


def getArgs():
    ap = ArgumentParser()
    ap.add_argument("session", type=str)  # folder containing cam_1/ and cam_2/
    ap.add_argument("-s", "--size", type=float, default=15)  # marker size in any unit
    ap.add_argument("-n", "--frames", type=int, default=21)  # frames taken per loop
    return ap.parse_args()


if __name__ == "__main__":
    args = getArgs()

    results = {}
    for cam in CAMS:
        fn = join(args.session, cam, "markers.json")
        if not isfile(fn):
            print(f"[Warning] {fn} not found, skipping")
            continue

        markers = loadMarkers(fn)
        missing = [l for l in LOOPS if l not in markers]
        if missing:
            print(f"[Warning] {cam}: missing {', '.join(missing)}, skipping")
            continue

        results[cam], counts = comparison_loops(markers, args.size, args.frames)
        short = {l: n for l, n in counts.items() if n < args.frames}
        if short:
            print(f"[Warning] {args.session} {cam}: fewer than {args.frames} frames {short}")

    out = join(args.session, "singles")
    makedirs(out, exist_ok=True)
    with open(join(out, "results.json"), "w") as w:
        json.dump(results, w)
