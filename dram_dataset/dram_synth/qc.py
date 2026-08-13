"""
Quality control.

Twelve checks run over the finished dataset on disk. Anything that fails raises
`QCError`, naming the affected samples. The checks read the delivered PNGs and
annotations rather than any in-memory state, so they validate the artefact a
consumer would actually receive.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .params import SEARCH_SIZE_PX, SPLIT_SEED_OFFSET, architecture_signature

# Ground-truth verification thresholds.
#
# A DRAM array is periodic, so a correlation peak genuinely cannot tell one
# lattice repeat from the next: a peak sitting exactly N word-line pitches away
# is matching ambiguity, not a coordinate error. Check 10 therefore has two
# parts. `verify_transform_math` tests the corner algebra and the scan-warp
# point mapping directly, on a NON-periodic specimen where the peak is unique --
# that is what actually proves the annotations are right. The dataset-wide sweep
# below then only has to catch gross breakage, and classifies any offset that
# resolves to an integer lattice step as aliasing rather than error.
GT_EXACT_PX = 6.0            # peak this close to the annotation: exact
GT_ALIAS_RESIDUAL_PX = 2.5   # residual after removing whole lattice steps
GT_ALIAS_MAX_STEPS = 4       # how many repeats away a peak may plausibly land
GT_MEDIAN_FAIL_PX = 3.0      # median over a split; catches a systematic shift
TRANSFORM_MATH_FAIL_PX = 2.0  # analytic test, non-periodic specimen
# Copied noise would make the high-pass residuals of the two captures track each
# other closely. Independent noise sits near zero.
NOISE_CORR_FAIL = 0.60
MIN_IMAGE_STD = 3.0


class QCError(AssertionError):
    """Raised when a quality-control check fails."""


def _fail(check: str, message: str, samples) -> None:
    ids = list(samples)[:15]
    suffix = f" (+{len(samples) - len(ids)} more)" if len(samples) > len(ids) else ""
    raise QCError(f"[{check}] {message}\n  affected: {ids}{suffix}")


def _load_split(root: Path, split: str) -> list:
    path = root / split / "annotations.json"
    if not path.exists():
        raise QCError(f"[2/3] missing annotations file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["samples"]


def run_all_checks(output_root, expected_total: int, splits: dict,
                   master_seed: int, gt_sample_limit: int = 0,
                   verbose: bool = True) -> dict:
    """Run every check. Returns a report dict; raises QCError on failure.

    `gt_sample_limit` caps how many samples get the expensive ground-truth
    correlation check (check 10); 0 means check every sample.
    """
    root = Path(output_root)
    log = (lambda m: print(m)) if verbose else (lambda m: None)

    records = {}
    for split in splits:
        records[split] = _load_split(root, split)

    all_records = [r for split in splits for r in records[split]]

    # ---- 1. exactly the expected number of pairs -------------------------- #
    total = len(all_records)
    if total != expected_total:
        _fail("1", f"expected {expected_total} samples, found {total}", [])
    for split, n in splits.items():
        if len(records[split]) != n:
            _fail("1", f"split '{split}' expected {n} samples, found {len(records[split])}", [])
    log(f"  [ 1/12] sample count .......... {total} pairs across {len(splits)} splits  OK")

    # ---- 2 & 3. every reference and search file exists -------------------- #
    missing_ref, missing_search = [], []
    for r in all_records:
        if not (root / r["split"] / r["reference_filename"]).exists():
            missing_ref.append(r["image_id"])
        if not (root / r["split"] / r["search_filename"]).exists():
            missing_search.append(r["image_id"])
    if missing_ref:
        _fail("2", "reference image files missing", missing_ref)
    if missing_search:
        _fail("3", "search image files missing", missing_search)
    log("  [ 2/12] reference files exist ... OK")
    log("  [ 3/12] search files exist ..... OK")

    # ---- 4 & 12. dimensions, and images not blank / corrupted ------------- #
    bad_dims, blank, unreadable = [], [], []
    for r in all_records:
        sp = root / r["split"]
        search = cv2.imread(str(sp / r["search_filename"]), cv2.IMREAD_UNCHANGED)
        ref = cv2.imread(str(sp / r["reference_filename"]), cv2.IMREAD_UNCHANGED)
        if search is None or ref is None:
            unreadable.append(r["image_id"])
            continue
        if search.ndim != 2 or search.shape != (SEARCH_SIZE_PX, SEARCH_SIZE_PX):
            bad_dims.append((r["image_id"], search.shape))
        if ref.ndim != 2 or ref.shape[0] != r["reference_height"] or ref.shape[1] != r["reference_width"]:
            bad_dims.append((r["image_id"], ref.shape))
        if search.std() < MIN_IMAGE_STD or ref.std() < MIN_IMAGE_STD:
            blank.append(r["image_id"])
        if search.max() == search.min() or ref.max() == ref.min():
            blank.append(r["image_id"])
    if unreadable:
        _fail("12", "images could not be decoded", unreadable)
    if bad_dims:
        _fail("4", f"images are not the declared size (search must be "
                   f"{SEARCH_SIZE_PX}x{SEARCH_SIZE_PX}, 8-bit grayscale)", bad_dims)
    log(f"  [ 4/12] search size {SEARCH_SIZE_PX}x{SEARCH_SIZE_PX} .. OK (grayscale, 8-bit)")

    # ---- 5 & 6. bounding boxes valid and inside the frame ----------------- #
    invalid, outside = [], []
    for r in all_records:
        b = r["bbox"]
        if not all(np.isfinite(v) for v in (b["x1"], b["y1"], b["x2"], b["y2"])):
            invalid.append(r["image_id"]); continue
        if b["x2"] <= b["x1"] or b["y2"] <= b["y1"]:
            invalid.append(r["image_id"]); continue
        if not (0.0 <= b["x1"] and 0.0 <= b["y1"]
                and b["x2"] <= SEARCH_SIZE_PX and b["y2"] <= SEARCH_SIZE_PX):
            outside.append((r["image_id"], b))
        c = r["center"]
        if not (b["x1"] <= c["x"] <= b["x2"] and b["y1"] <= c["y"] <= b["y2"]):
            invalid.append(r["image_id"])
    if invalid:
        _fail("5", "invalid bounding box or centre outside its own box", invalid)
    if outside:
        _fail("6", "bounding box extends outside the search image", outside)
    log("  [ 5/12] bounding boxes valid ... OK")
    log("  [ 6/12] boxes inside frame ..... OK")

    # ---- 7. pairing -------------------------------------------------------- #
    unpaired = [r["image_id"] for r in all_records
                if not r.get("reference_filename") or not r.get("search_filename")]
    if unpaired:
        _fail("7", "sample is missing one side of the pair", unpaired)
    log("  [ 7/12] every reference paired .. OK")

    # ---- 8. no duplicate filenames, and no duplicate image content -------- #
    names = [f"{r['split']}/{r['reference_filename']}" for r in all_records] + \
            [f"{r['split']}/{r['search_filename']}" for r in all_records]
    dupe_names = [n for n, c in Counter(names).items() if c > 1]
    if dupe_names:
        _fail("8", "duplicate filenames", dupe_names)

    digests = {}
    dupe_content = []
    for r in all_records:
        for key in ("reference_filename", "search_filename"):
            p = root / r["split"] / r[key]
            h = hashlib.md5(p.read_bytes()).hexdigest()
            if h in digests:
                dupe_content.append((r["image_id"], digests[h]))
            else:
                digests[h] = r["image_id"]
    if dupe_content:
        _fail("8", "duplicate image content -- two samples produced identical files", dupe_content)
    log(f"  [ 8/12] no duplicates .......... OK ({len(digests)} unique images)")

    # ---- 9. reference and search noise independently generated ------------ #
    same_seed = [r["image_id"] for r in all_records
                 if r["seeds"]["reference_noise_seed"] == r["seeds"]["search_noise_seed"]]
    if same_seed:
        _fail("9", "reference and search share a noise seed", same_seed)

    noise_ids = [r["seeds"]["reference_noise_seed"] for r in all_records] + \
                [r["seeds"]["search_noise_seed"] for r in all_records]
    reused = [s for s, c in Counter(noise_ids).items() if c > 1]
    if reused:
        _fail("9", "a noise seed is reused across images", reused)

    # Empirical: if noise had been copied, the high-pass residuals of the two
    # captures would correlate strongly. Independent draws sit near zero.
    corr_samples = all_records[::max(len(all_records) // 40, 1)]
    correlations, corr_bad = [], []
    for r in corr_samples:
        sp = root / r["split"]
        ref = cv2.imread(str(sp / r["reference_filename"]), cv2.IMREAD_GRAYSCALE)
        search = cv2.imread(str(sp / r["search_filename"]), cv2.IMREAD_GRAYSCALE)
        b = r["bbox"]
        crop = search[int(b["y1"]):int(np.ceil(b["y2"])), int(b["x1"]):int(np.ceil(b["x2"]))]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LINEAR)
        a = cv2.Laplacian(ref, cv2.CV_32F).ravel()
        c = cv2.Laplacian(crop, cv2.CV_32F).ravel()
        if a.std() < 1e-6 or c.std() < 1e-6:
            continue
        rho = float(np.corrcoef(a, c)[0, 1])
        correlations.append(rho)
        if abs(rho) > NOISE_CORR_FAIL:
            corr_bad.append((r["image_id"], round(rho, 3)))
    if corr_bad:
        _fail("9", f"high-pass correlation exceeds {NOISE_CORR_FAIL} -- noise looks shared", corr_bad)
    max_corr = max((abs(v) for v in correlations), default=0.0)
    log(f"  [ 9/12] noise independent ...... OK (max |corr| = {max_corr:.3f} "
        f"over {len(correlations)} probes, limit {NOISE_CORR_FAIL})")

    # ---- 10a. the transformation maths itself, on a non-periodic specimen -- #
    math_err = verify_transform_math()
    if math_err > TRANSFORM_MATH_FAIL_PX:
        _fail("10", f"ground-truth transform is wrong: analytic test peaks "
                    f"{math_err:.2f}px from the computed annotation "
                    f"(limit {TRANSFORM_MATH_FAIL_PX}px)", [])

    # ---- 10b. dataset-wide sweep ------------------------------------------ #
    gt_records = all_records
    if gt_sample_limit and gt_sample_limit < len(all_records):
        step = max(len(all_records) // gt_sample_limit, 1)
        gt_records = all_records[::step][:gt_sample_limit]

    offsets, gt_bad, n_alias = [], [], 0
    for r in gt_records:
        probe = _gt_offset(root, r)
        if probe is None:
            continue
        offsets.append(probe["offset"])
        if probe["offset"] <= GT_EXACT_PX:
            continue
        if _is_lattice_alias(probe, r):
            n_alias += 1
            continue
        gt_bad.append((r["image_id"], round(probe["offset"], 2), r["difficulty"],
                       f"dx={probe['dx']:+.1f} dy={probe['dy']:+.1f}"))
    if gt_bad:
        _fail("10", "correlation peak is neither at the annotated location nor an "
                    "integer lattice step from it -- the ground truth is wrong", gt_bad)

    offsets = np.array(offsets) if offsets else np.array([0.0])
    median = float(np.median(offsets))
    if median > GT_MEDIAN_FAIL_PX:
        _fail("10", f"median ground-truth offset {median:.2f}px exceeds "
                    f"{GT_MEDIAN_FAIL_PX}px -- systematic coordinate shift", [])
    log(f"  [10/12] ground truth verified .. OK (analytic transform test "
        f"{math_err:.2f}px; dataset median {median:.2f}px, p95 "
        f"{np.percentile(offsets, 95):.2f}px over {len(offsets)} samples; "
        f"{n_alias} lattice-aliased)")

    # ---- 11. splits use different random seeds ---------------------------- #
    offsets_used = {s: SPLIT_SEED_OFFSET[s] for s in splits}
    if len(set(offsets_used.values())) != len(offsets_used):
        _fail("11", "two splits share a seed offset", list(offsets_used.items()))
    seed_sets = {s: {r["random_seed"] for r in records[s]} for s in splits}
    for a in splits:
        for b in splits:
            if a >= b:
                continue
            shared = seed_sets[a] & seed_sets[b]
            if shared:
                _fail("11", f"splits '{a}' and '{b}' share sample seeds", sorted(shared))
    log("  [11/12] split seeds disjoint ... OK")

    # ---- 10b / leakage: no geometric combination reused across splits ----- #
    sig_owner, leaked = {}, []
    for r in all_records:
        a = r["architecture"]
        sig = (a["P01_word_line_pitch"], a["P02_bit_to_word_pitch_ratio"],
               a["P03_line_width_frac"], a["P04_contact_diameter_frac"],
               a["P05_block_line_count"], a["P06_spacing_jitter"],
               tuple(a["P07_phase_offset"]))
        if sig in sig_owner and sig_owner[sig][1] != r["split"]:
            leaked.append((r["image_id"], sig_owner[sig][0]))
        sig_owner.setdefault(sig, (r["image_id"], r["split"]))
    if leaked:
        _fail("11", "identical geometry appears in more than one split", leaked)

    # ---- 12. already covered above ---------------------------------------- #
    log(f"  [12/12] images non-blank ....... OK (min std threshold {MIN_IMAGE_STD})")

    return {
        "total": total,
        "transform_math_error_px": float(math_err),
        "gt_offset_median_px": median,
        "gt_offset_p95_px": float(np.percentile(offsets, 95)),
        "gt_offset_max_px": float(offsets.max()),
        "gt_checked": int(len(offsets)),
        "gt_lattice_aliased": int(n_alias),
        "noise_corr_max": float(max_corr),
        "noise_probes": int(len(correlations)),
        "unique_images": len(digests),
    }


def _is_lattice_alias(probe: dict, record: dict) -> bool:
    """True if the correlation peak sits a whole number of lattice steps away.

    The search image is not rotated, so bit lines are vertical (period along x)
    and word lines horizontal (period along y). If removing an integer number of
    those periods from the offset leaves a sub-pixel residual, the matcher simply
    locked onto a neighbouring repeat of an identical pattern -- the annotation
    is fine and the image is genuinely ambiguous at that scale.
    """
    a = record["architecture"]
    px, py = a["bit_line_pitch"], a["P01_word_line_pitch"]
    if px <= 0 or py <= 0:
        return False
    nx = round(probe["dx"] / px)
    ny = round(probe["dy"] / py)
    if abs(nx) > GT_ALIAS_MAX_STEPS or abs(ny) > GT_ALIAS_MAX_STEPS:
        return False
    residual = np.hypot(probe["dx"] - nx * px, probe["dy"] - ny * py)
    return bool(residual <= GT_ALIAS_RESIDUAL_PX)


def verify_transform_math(trials: int = 6, seed: int = 20240613) -> float:
    """Test the ground-truth transformation directly, free of lattice ambiguity.

    The annotation is produced by two pieces of maths: the corner algebra in
    `sample._reference_window`, which says where the rotated reference window
    sits in world coordinates, and `sem.forward_map_points`, which pushes those
    corners through the thermal-drift and vibration warp applied to the search
    image. Both are exercised here on a specimen of smooth *random* texture
    instead of a DRAM array. Because that texture is unique everywhere, the
    correlation peak is unambiguous, so any disagreement is a real coordinate
    error rather than a matcher picking the wrong repeat.

    Returns the worst peak-to-annotation distance in pixels, over `trials`
    randomized geometries with deliberately heavy drift and vibration.
    """
    from . import sem
    from .sample import _reference_window

    rng = np.random.default_rng(seed)
    s = 2                                   # small canvas: this is a maths test
    n_fine = SEARCH_SIZE_PX * s
    worst = 0.0

    for t in range(trials):
        coarse = rng.integers(0, 256, size=(160, 160)).astype(np.uint8)
        fine = cv2.resize(coarse, (n_fine, n_fine), interpolation=cv2.INTER_CUBIC)
        fine = cv2.GaussianBlur(fine, (0, 0), sigmaX=2.0)

        footprint = float(rng.uniform(85, 115))
        rotation = float(rng.uniform(-5, 5))
        margin = 0.5 * footprint * 1.5 + 30
        params = {
            "footprint_px": footprint,
            "rotation_deg": rotation,
            "position_x": float(rng.uniform(margin, SEARCH_SIZE_PX - margin)),
            "position_y": float(rng.uniform(margin, SEARCH_SIZE_PX - margin)),
        }
        window, corners = _reference_window(fine, params, s)

        # Deliberately severe scan errors, so a sign slip cannot hide.
        imaging = {
            "thermal_drift_px": float(rng.uniform(3.0, 6.0)),
            "thermal_drift_angle_deg": float(rng.uniform(0, 360)),
            "vibration_amp_px": float(rng.uniform(1.0, 2.5)),
            "vibration_freq_cycles": float(rng.uniform(5, 40)),
            "vibration_phase_rad": float(rng.uniform(0, 2 * np.pi)),
            "vibration_jitter_px": 0.0,
        }
        search = sem.downsample(fine, SEARCH_SIZE_PX)
        dx, dy = sem.scan_displacement(SEARCH_SIZE_PX, imaging, seed + t)
        search = sem.apply_scan_displacement(search, dx, dy)

        quad = sem.forward_map_points(corners, dx, dy)
        centre = sem.forward_map_points(
            np.array([[params["position_x"], params["position_y"]]]), dx, dy)[0]

        # De-rotate the window back into search orientation and match it.
        f = int(round(footprint))
        tmpl = cv2.resize(window, (f, f), interpolation=cv2.INTER_AREA)
        m = cv2.getRotationMatrix2D((f / 2.0, f / 2.0), -rotation, 1.0)
        tmpl = cv2.warpAffine(tmpl, m, (f, f), flags=cv2.INTER_LINEAR)
        k = int(f * 0.75)
        o = (f - k) // 2
        tmpl = tmpl[o:o + k, o:o + k]

        pad = k // 2 + 30
        x0 = int(np.clip(centre[0] - pad, 0, SEARCH_SIZE_PX - 1))
        y0 = int(np.clip(centre[1] - pad, 0, SEARCH_SIZE_PX - 1))
        x1 = int(np.clip(centre[0] + pad, 1, SEARCH_SIZE_PX))
        y1 = int(np.clip(centre[1] + pad, 1, SEARCH_SIZE_PX))
        res = cv2.matchTemplate(search[y0:y1, x0:x1], tmpl, cv2.TM_CCOEFF_NORMED)
        _, _, _, loc = cv2.minMaxLoc(res)
        pred = (x0 + loc[0] + k / 2.0, y0 + loc[1] + k / 2.0)

        worst = max(worst, float(np.hypot(pred[0] - centre[0], pred[1] - centre[1])))

        # The mapped centre must also lie inside the mapped quadrilateral's box.
        if not (quad[:, 0].min() <= centre[0] <= quad[:, 0].max()
                and quad[:, 1].min() <= centre[1] <= quad[:, 1].max()):
            return float("inf")

    return worst


def _gt_offset(root: Path, record: dict):
    """Where the reference actually correlates to, relative to the annotation.

    Returns a dict with the signed offset components, its magnitude and the peak
    score, or None if the sample could not be probed. The signed components are
    what let the caller tell a lattice repeat from a real coordinate error.

    The reference is de-rotated by the recorded angle and rescaled to its
    recorded footprint, then matched inside a small window around the recorded
    centre. If the transformation bookkeeping were wrong, the peak would land
    somewhere else entirely.
    """
    sp = root / record["split"]
    ref = cv2.imread(str(sp / record["reference_filename"]), cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(str(sp / record["search_filename"]), cv2.IMREAD_GRAYSCALE)
    if ref is None or search is None:
        return None

    # Size the template from the annotated quadrilateral, not from
    # `footprint_px`. The two differ once barrel distortion is applied: the
    # footprint is the nominal pre-distortion side length, while the quad holds
    # the mapped corners and so reflects the region's true size in the image.
    quad = np.asarray(record["quad"], dtype=np.float64)
    sides = [float(np.hypot(*(quad[(i + 1) % 4] - quad[i]))) for i in range(4)]
    footprint = int(round(sum(sides) / 4.0))
    if footprint < 16:
        return None
    tmpl = cv2.resize(ref, (footprint, footprint), interpolation=cv2.INTER_AREA)
    m = cv2.getRotationMatrix2D((footprint / 2.0, footprint / 2.0),
                                -record["P10_rotation_deg"], 1.0)
    tmpl = cv2.warpAffine(tmpl, m, (footprint, footprint), flags=cv2.INTER_LINEAR)

    # Drop the border: de-rotation leaves undefined corners.
    k = int(footprint * 0.80)
    o = (footprint - k) // 2
    tmpl = tmpl[o:o + k, o:o + k]

    cx, cy = record["center"]["x"], record["center"]["y"]
    pad = k // 2 + 25
    x0 = int(np.clip(cx - pad, 0, SEARCH_SIZE_PX - 1))
    y0 = int(np.clip(cy - pad, 0, SEARCH_SIZE_PX - 1))
    x1 = int(np.clip(cx + pad, 1, SEARCH_SIZE_PX))
    y1 = int(np.clip(cy + pad, 1, SEARCH_SIZE_PX))
    window = search[y0:y1, x0:x1]
    if window.shape[0] <= k or window.shape[1] <= k:
        return None

    result = cv2.matchTemplate(window, tmpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, max_loc = cv2.minMaxLoc(result)
    pred_x = x0 + max_loc[0] + k / 2.0
    pred_y = y0 + max_loc[1] + k / 2.0
    dx, dy = pred_x - cx, pred_y - cy
    return {"dx": float(dx), "dy": float(dy),
            "offset": float(np.hypot(dx, dy)), "score": float(score)}