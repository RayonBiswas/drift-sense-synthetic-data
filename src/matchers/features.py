"""Local feature matching (SIFT / ORB). The fallback.

Keypoint matching is the standard answer to rotation, scale and partial
occlusion: describe a few distinctive points invariantly and let RANSAC find
the transform that explains the most correspondences. It is included here
because it is the obvious thing a reviewer will ask about -- and because on
this data it is expected to struggle, which is itself a result worth showing.

Why it struggles: the reference occupies roughly 100x100 px of the search
image, so there is very little to describe; and a memory array is built from
one motif repeated on a lattice, so the descriptors that do exist are nearly
identical to each other and Lowe's ratio test throws most of them away.
Both images are optionally upsampled to give the detector more to work with.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from .common import INNER_FRACTION, MatchResult, make_template, zncc_score_at

RATIO = 0.8
MIN_MATCHES = 4


def _detector(kind: str):
    kind = kind.lower()
    if kind == "sift":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("this OpenCV build has no SIFT")
        return cv2.SIFT_create(nfeatures=1200), cv2.NORM_L2
    if kind == "orb":
        return cv2.ORB_create(nfeatures=2000), cv2.NORM_HAMMING
    if kind == "akaze":
        return cv2.AKAZE_create(), cv2.NORM_HAMMING
    raise ValueError(f"unknown detector: {kind}")


def feature_match(reference: np.ndarray, search: np.ndarray,
                  detector: str = "sift", base_size: int = 100,
                  upsample: float = 2.0, method_name: str = None) -> MatchResult:
    """Detect, match with a ratio test, fit a similarity transform by RANSAC."""
    t0 = time.perf_counter()
    method_name = method_name or f"features_{detector}"
    det, norm = _detector(detector)

    template = make_template(reference, int(base_size), 0.0, INNER_FRACTION)
    if upsample != 1.0:
        tpl_up = cv2.resize(template, None, fx=upsample, fy=upsample, interpolation=cv2.INTER_CUBIC)
        srch_up = cv2.resize(search, None, fx=upsample, fy=upsample, interpolation=cv2.INTER_CUBIC)
    else:
        tpl_up, srch_up = template, search

    kp1, des1 = det.detectAndCompute(tpl_up, None)
    kp2, des2 = det.detectAndCompute(srch_up, None)

    def _fail(reason: str) -> MatchResult:
        return MatchResult(
            x=float(search.shape[1] / 2.0), y=float(search.shape[0] / 2.0),
            score=-1.0, method=method_name, size=int(base_size), n_candidates=0,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            extra={"failed": True, "reason": reason, "native_metric": "inliers",
                   "native_score": 0.0, "keypoints_ref": len(kp1 or []),
                   "keypoints_search": len(kp2 or [])},
        )

    if des1 is None or des2 is None or len(kp1) < MIN_MATCHES or len(kp2) < MIN_MATCHES:
        return _fail("too few keypoints")

    matcher = cv2.BFMatcher(norm)
    pairs = matcher.knnMatch(des1, des2, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < RATIO * n.distance]
    if len(good) < MIN_MATCHES:
        return _fail(f"only {len(good)} matches survived the ratio test")

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    matrix, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                  ransacReprojThreshold=3.0)
    if matrix is None:
        return _fail("RANSAC found no consistent transform")

    k = tpl_up.shape[0]
    centre = np.array([[k / 2.0, k / 2.0, 1.0]], dtype=np.float32).T
    mapped = matrix @ centre
    cx = float(mapped[0, 0]) / upsample
    cy = float(mapped[1, 0]) / upsample

    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    angle = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    n_in = int(inliers.sum()) if inliers is not None else 0

    return MatchResult(
        x=cx, y=cy, score=zncc_score_at(search, template, cx, cy),
        method=method_name, angle=angle, size=int(round(base_size * scale)),
        n_candidates=1,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        extra={"native_metric": "inliers", "native_score": float(n_in),
               "matches": len(good), "inliers": n_in,
               "keypoints_ref": len(kp1), "keypoints_search": len(kp2),
               "est_scale": scale, "est_angle": angle},
    )
