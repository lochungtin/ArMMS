import json
from itertools import combinations as comb

import numpy as np
from natsort import natsorted

from src.marker import Marker


def loadMarkers(fn):
    # markers.json: {group: {image: {marker_id: [[x, y] * 4]}}}
    with open(fn, "r") as r:
        raw = json.load(r)

    markers = {}
    for grpName, grp in raw.items():
        markers[grpName] = {}
        for imgName, img in grp.items():
            _m = {mID: Marker(v) for mID, v in img.items()}
            [__m.edgeLength() for __m in _m.values()]
            markers[grpName][imgName] = _m

    return markers


def displacement(m0, m1, MARKER_SIZE):
    # midpoint displacement scaled from pixels to marker units
    [dx, dy], d = m0 - m1
    r = MARKER_SIZE / ((m0.edgeLength() + m1.edgeLength()) / 2)
    return r * d, r * dx, r * dy


def compareAll(grp, MARKER_SIZE, log=None):
    # every pair of images in the group, matching marker ids
    results = {}
    for i0, i1 in comb(grp.keys(), 2):
        s0, s1 = set(grp[i0].keys()), set(grp[i1].keys())
        markers = sorted(list(s0.intersection(s1)))

        if log:
            log(f"Matching pairs found: {' '.join(markers)}")

        for mID in markers:
            d, dx, dy = displacement(grp[i0][mID], grp[i1][mID], MARKER_SIZE)

            if mID not in results:
                results[mID] = {}

            results[mID][f"{i0}/{i1}"] = {"d": d, "dx": dx, "dy": dy}

    return results


def compareSeries(grp, MARKER_SIZE):
    # first image (natural sort) against every image, nan where a marker is missing
    markers = set()
    for _markers in grp.values():
        for mID in _markers.keys():
            markers.add(mID)

    imgs = natsorted(grp.keys())
    ref = imgs[0]

    results = {}
    for img in imgs:
        for mID in sorted(list(markers)):
            if mID not in results:
                results[mID] = []
            m0, m1 = grp[ref].get(mID), grp[img].get(mID)
            if m0 is None or m1 is None:
                results[mID].append([np.nan, np.nan, np.nan])
            else:
                results[mID].append(list(displacement(m0, m1, MARKER_SIZE)))

    return results
