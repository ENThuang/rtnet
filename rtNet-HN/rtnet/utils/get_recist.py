import numpy as np
import json
import os
import time
import sys
import os.path as osp
import cv2
from collections import OrderedDict
from scipy.spatial.distance import pdist, squareform


def mask2contour(mask):
    mask = mask.astype("uint8")
    tmp = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours1 = tmp[0]
    contour = contours1[
        np.argmax([len(contour1) for contour1 in contours1])
    ]  # only keep the longest countour
    contour = contour.squeeze().astype("float32")
    if contour.ndim == 1:
        contour = contour[None, :]
    return contour


def find_recist_in_contour(contour):
    if len(contour) == 1:
        return np.array([0, 0]), np.array([0, 0]), 1, 1
    D = squareform(pdist(contour)).astype("float32")
    long_diameter_len = D.max()
    endpt_idx = np.where(D == long_diameter_len)
    endpt_idx = np.array([endpt_idx[0][0], endpt_idx[1][0]])
    long_diameter_vec = (contour[endpt_idx[0]] - contour[endpt_idx[1]])[:, None]

    side1idxs = np.arange(endpt_idx.min(), endpt_idx.max() + 1)
    side2idxs = np.hstack(
        (np.arange(endpt_idx.min() + 1), np.arange(endpt_idx.max(), len(contour)))
    )
    perp_diameter_lens = np.empty((len(side1idxs),), dtype=float)
    perp_diameter_idx2s = np.empty((len(side1idxs),), dtype=int)
    for i, idx1 in enumerate(side1idxs):
        short_diameter_vecs = contour[side2idxs] - contour[idx1]
        dot_prods_abs = np.abs(np.matmul(short_diameter_vecs, long_diameter_vec))
        idx2 = np.where(dot_prods_abs == dot_prods_abs.min())[0]
        if len(idx2) > 1:
            idx2 = idx2[np.sum(short_diameter_vecs[idx2] ** 2, axis=1).argmax()]
        idx2 = side2idxs[idx2]  # find the one that is perpendicular with long axis
        perp_diameter_idx2s[i] = idx2
        perp_diameter_lens[i] = D[idx1, idx2]
    short_diameter_len = perp_diameter_lens.max()
    short_diameter_idx1 = side1idxs[perp_diameter_lens.argmax()]
    short_diameter_idx2 = perp_diameter_idx2s[perp_diameter_lens.argmax()]
    short_diameter = np.array([short_diameter_idx1, short_diameter_idx2])
    long_diameter = endpt_idx

    return long_diameter, short_diameter, long_diameter_len, short_diameter_len


def get_RECIST(mask, spacing):
    areas_2D = mask.sum(axis=(1, 2))
    slices = np.where(areas_2D > 0)[0]
    short_diameter_lens = []
    for slice in slices:
        contour = mask2contour(mask[slice])
        (
            long_diameter,
            short_diameter,
            long_diameter_len,
            short_diameter_len,
        ) = find_recist_in_contour(contour)
        short_diameter_lens.append(short_diameter_len)
    short_diameter_len = max(short_diameter_lens)
    return short_diameter_len * spacing


if __name__ == "__main__":
    # mask: a D-by-H-by-W numpy array, contains a single LN mask
    # spacing: mm-per-pixel in the xy plane
    # get_RECIST(mask, spacing)
    raise NotImplementedError
