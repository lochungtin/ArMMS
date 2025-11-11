import cv2 as cv
import numpy as np
from PIL import Image


class Marker:
    EDGE_MAP = [[0, 1], [1, 2], [2, 3], [3, 0]]

    def __init__(self, vertices):
        self.vertices = np.array(vertices)
        self.midpoint = self.vertices.mean(axis=0).astype(int)
        self.normalised = (self.vertices - self.midpoint).astype(int)
        self.edges = [
            np.linalg.norm(self.vertices[a] - self.vertices[b]) for a, b in Marker.EDGE_MAP
        ]

        self.edgeL = None
        self.rMat = None

    def create(l):
        return Marker([[0, 0], [l, 0], [l, l], [0, l]])

    def __sub__(self, o):
        dVec = self.midpoint - o.midpoint
        return dVec, np.linalg.norm(dVec)

    def tolist(self):
        return self.vertices.tolist()

    def addZ(self):
        return Marker()

    def dropZ(self):
        return Marker(self.vertices @ np.array([[1, 0], [0, 1], [0, 0]]))

    def scale(self, scale):
        return Marker(self.vertices * scale)

    def rotate(self, M):
        nV3d = self.vertices @ np.array([[1, 0, 0], [0, 1, 0]])
        nV2d = nV3d @ M.T @ np.array([[1, 0], [0, 1], [0, 0]])
        return Marker(nV2d)

    def loadTextures(self, filename, size=-1):
        template = ~cv.imread(filename)

        if size == -1:
            size = max(np.abs(self.normalised).flatten()) * 2
        radius = size // 2

        canonShp = np.float32(Marker.create(template.shape[0]).vertices)
        targetShp = np.float32(self.normalised + radius)
        M = cv.getPerspectiveTransform(canonShp, targetShp)
        img = ~cv.warpPerspective(template, M, (size, size)).astype(np.uint8)
        return Image.fromarray(img)

    def fillWhite(self, size=-1):
        if size == -1:
            size = max(np.abs(self.normalised).flatten()) * 2
        radius = size // 2
        base = np.zeros((size, size, 3))
        return cv.fillPoly(base, pts=[self.normalised + radius], color=(255, 255, 255))

    def edgeLength(self):
        if self.edgeL == None:

            def step(a, b):
                A = b[0] ** 2 + b[1] ** 2 - a[0] ** 2 - a[1] ** 2
                B = -a[0] * b[0] - a[1] * b[1]
                z0 = np.sqrt((A + np.sqrt(A**2 + (4 * B**2))) / 2)
                z1 = B / z0
                p0 = np.float32([a[0], a[1], z0])
                p1 = np.float32([b[0], b[1], z1])
                return np.linalg.norm(p0 - p1)

            self.edgeL = np.mean(
                [step(self.normalised[a], self.normalised[b]) for a, b in Marker.EDGE_MAP]
            )

        return self.edgeL

    def recoverMatrix(self):
        if self.rMat == None:

            self.edgeLength()
            lp, ln = self.edgeL // 2, -self.edgeL // 2
            A = [
                [ln, ln, 0, 0],
                [0, 0, ln, ln],
                [lp, ln, 0, 0],
                [0, 0, lp, ln],
                [lp, lp, 0, 0],
                [0, 0, lp, lp],
                [ln, lp, 0, 0],
                [0, 0, ln, lp],
            ]
            b = self.vertices.flatten().reshape((-1, 1))
            r = np.linalg.lstsq(np.array(A), b, rcond=None)[0].flatten()

            self.rMat = np.zeros((3, 3))
            self.rMat[0:2, 0:2] = r.reshape(2, 2)
            self.rMat[0, 2] = np.sqrt(max(0, 1 - r[0] ** 2 - r[1] ** 2))
            self.rMat[1, 2] = np.sqrt(max(0, 1 - r[2] ** 2 - r[3] ** 2))
            self.rMat[2] = np.cross(self.rMat[0], self.rMat[1])

        return self.rMat

    def __str__(self):
        return str(self.vertices)

    def __repr__(self):
        return self.__str__()
