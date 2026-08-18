# DRAM Synthetic Dataset — Reference-to-Search Image Localization

A procedurally generated dataset of 1,000 SEM-style image pairs for the
Drift-Sense semiconductor image-matching problem. Every sample is a
high-magnification **reference** image of a small area of a DRAM array, a
low-magnification **search** image of the surrounding device, and the exact
coordinates of the reference inside the search image.

Nothing is copied from a real image. The device geometry, the imaging physics
and the ground truth are all computed.

---

## 1. What the dataset represents

A tool has a stored reference image of a small landmark on a wafer, and a fresh
wide-field scan. It has to find where the landmark is in the new scan. That is
hard because the two pictures are separate physical captures: different
magnification, different dose, different noise, and a stage that drifts and
vibrates while each frame is being rastered.

Each sample therefore contains:

| | |
|---|---|
| **Reference image** | 100–256 px square (randomized), high magnification, cut from the fine specimen canvas |
| **Search image** | exactly **1000 × 1000 px**, 8-bit grayscale |
| **Ground truth** | bounding box + centre of the reference inside the search image, plus the exact rotated quadrilateral |
| **Metadata** | all 20 generation parameters, seeds, defect counts and difficulty |

The reference occupies roughly **100 × 100 px** of the search image.

### A note on "10× magnification"

Three of the spec's constraints cannot all hold at once: a reference stored at
100–256 px, a footprint of ~100 px in the search image, and a 10× magnification
ratio between them. Any two are satisfiable; all three are not. This generator
keeps the two that define the task:

* the search image is exactly 1000 × 1000, and
* the reference's footprint in it is ~100 × 100 px — which is what the ground
  truth box means.

The **10× reduction is real and is in the rendering path**: the specimen is
rasterized onto a 10000 × 10000 fine canvas and the search image is that canvas
area-averaged down by 10×. The reference is cut from the *un-downsampled* canvas,
so it carries genuinely finer detail than the search image. What comes out is a
stored-resolution ratio of about 1.0–2.6× rather than 10×. Set `--supersample`
to change the canvas reduction factor.

---

## 2. How the DRAM pattern is generated

Everything is drawn in **world units**, where one world unit = one pixel of the
final search image, then rasterized at `--supersample` (default 10) times that
density and area-averaged down. Drawing supersampled and reducing is what gives
clean anti-aliased edges instead of staircased ones.

The layout is built bottom-up in `dram_synth/layout.py`:

1. **Peripheral / routing strips** — flat mid-grey material with sparse
   orthogonal interconnect, filling the whole canvas as a base layer.
2. **Array blocks ("mats")** — the periodic cell array, tiled across the canvas
   with strips between them. Each mat gets its **own lattice phase and a ±2 %
   pitch trim**, which is what makes one region of the search image
   distinguishable from another. Without it the canvas would be one perfect
   lattice and localization would be genuinely ill-posed.
3. **Word lines** — periodic horizontal lines.
4. **Bit lines** — periodic vertical lines at ~1.3–1.8× the word-line pitch (the
   6F² folded-bitline proportion).
5. **Storage-node contacts** — dots on a *checkerboard* of the line
   intersections, one per two cells, as in a real folded-bitline layout rather
   than a naive dot at every crossing.

Line positions carry a per-line random walk (line-placement error) and each line
gets its own width perturbation (CD variation), so no two lines are identical
and no lattice is perfect.

**Defects are injected during rasterization**, not painted on afterwards. They
are therefore properties of the *device* and appear consistently in both images —
unlike noise and blur, which are properties of each *capture*. Defect modes:
broken line, missing contact, displaced contact, line-width variation,
contamination particle, local intensity anomaly, and a locally out-of-focus
patch. Defect density is 0 for most easy samples and rises with difficulty, so
defects are **not** present in every image.

---

## 3. How degradation is applied

The reference and the search run through the same chain in `dram_synth/sem.py`
**separately**, with independently drawn parameters and independently seeded
random generators. They are two different photographs of the same object.

```
specimen
  1. edge brightening      secondary-electron yield rises at feature edges  (P14)
  2. downsample            detector sampling of the supersampled specimen
  3. thermal drift         slow monotonic displacement over the frame       (P19)
  4. vibration             quasi-periodic row displacement + jitter         (P20)
  5. optical blur          beam spot / defocus                              (P11)
  6. shading               low-frequency collection-efficiency variation    (P15)
  7. contrast / brightness detector gain and offset                    (P13/P12)
  8. sensor noise          Poisson shot noise + Gaussian read noise         (P16)
```

**Edge brightening** is not a sharpening filter. A morphological gradient finds
where material boundaries geometrically *are*, that map is softened, and it is
then **added** as extra emitted signal. Feature interiors are untouched. The
physical story is that more secondary electrons escape near an edge, so edges
read brighter than the flat top of the same feature.

**Thermal drift (P19)** models a stage that expands as it warms. A raster scan
builds an image row by row, so the displacement accumulates monotonically down
the frame — modelled as `0.75t + 0.25t²` of the total drift, in a random
direction: mostly linear with a slight acceleration.

**Vibration (P20)** models floor, pump and coolant-line coupling into the
column: a sinusoidal row displacement at a fixed frequency (3–60 cycles per
frame), plus an incoherent row-to-row jitter term for broadband mechanical
noise. The slow-scan axis gets a weaker component at a different beat.

**Noise is never shared.** The reference and search draw from separate
`numpy` generators seeded from separate derived seeds; there is no noise array
anywhere in the codebase that both images touch. QC check 9 verifies this both
structurally (seeds are distinct and never reused) and empirically (the
high-pass residuals of the two captures are uncorrelated).

Search images are drawn from consistently harsher ranges than reference images —
the search is the fast, wide-area scan.

---

## 4. How ground truth is calculated

**The ground truth is never measured from the output images.** It is computed
analytically and pushed through the same transformations the image gets.

1. The reference window is a square of side `F = 100 × scale` world units,
   centred at `(P17, P18)` and rotated by `P10`. Its four corners in world
   coordinates are `centre + A⁻¹ · (±F/2, ±F/2)`, where `A` is the linear part
   of the same OpenCV rotation matrix used to cut the window — so the algebra
   and the pixels cannot disagree.
2. World coordinates equal search-image pixels by construction (the canvas
   reduction is an exact integer factor), so no rescaling step is needed.
3. The search image is then warped by thermal drift and vibration. Those are
   per-row displacements `dx[r], dy[r]`; `sem.forward_map_points` pushes the four
   corners through the *same* functions, solving `r + dy[r] = Y` by fixed-point
   iteration.
4. `bbox` is the axis-aligned bounding box of those four mapped corners.
   `quad` stores the exact rotated corners, and `center` is the mapped centre.

Because the box is axis-aligned around a rotated square, it is slightly larger
than `F` — about 1.09 × F at the maximum 5° rotation. Use `quad` if you need the
exact region.

**Reference sourcing.** The reference is *cut from* the search's own layout
rather than pasted into it. Pasting a foreign patch leaves a seam and makes the
surrounding context inconsistent with the patch; cutting gives a seam-free
image and a ground truth that is exact by construction.

---

## 5. Quick start

```bash
pip install -r requirements.txt
```

### Generate 30 samples (fast check of the generator)

```bash
python generate_dataset.py --num-samples 30 --output debug_dataset --seed 42
```

Takes well under a minute and runs the full QC suite. Use this before committing
to the full run.

### Generate 1,000 samples

```bash
python generate_dataset.py --num-samples 1000 --output dataset --seed 42
```

About 4 minutes on 6 workers, and roughly **800 MB** on disk.

### Options

| flag | default | meaning |
|---|---|---|
| `--num-samples` | 1000 | total pairs, split 80 / 10 / 10 |
| `--output` | `dataset` | output root |
| `--seed` | 42 | master seed |
| `--workers` | auto | processes; auto = `cpu_count-1` capped at 6, `1` = serial |
| `--supersample` | 10 | fine-canvas factor; 6 is ~2× faster and uses ~⅓ the memory |
| `--skip-qc` | off | skip the 12 checks |
| `--qc-gt-limit` | 0 | cap samples given the ground-truth correlation check (0 = all) |

Generation is embarrassingly parallel — samples share no state — so `--workers`
scales nearly linearly. Each worker holds one `1000 × supersample` square canvas,
so at `--supersample 10` budget ~250 MB per worker.

---

## 6. Directory structure

```
dataset/
├── train/
│   ├── references/       sample_000001.png … sample_000800.png
│   ├── searches/         sample_000001.png … sample_000800.png
│   └── annotations.json
├── validation/
│   ├── references/       sample_000801.png … sample_000900.png
│   ├── searches/
│   └── annotations.json
├── test/
│   ├── references/       sample_000901.png … sample_001000.png
│   ├── searches/
│   └── annotations.json
└── metadata/
    └── generation_config.json
```

Image ids run continuously across the whole dataset, so no id or filename can
collide between splits. All images are 8-bit grayscale PNG.

---

## 7. Annotation format

`annotations.json` per split, with a `samples` array:

```json
{
  "image_id": "sample_000123",
  "split": "train",
  "reference_filename": "references/sample_000123.png",
  "search_filename": "searches/sample_000123.png",
  "reference_width": 196, "reference_height": 196,
  "search_width": 1000,   "search_height": 1000,

  "bbox":   { "x1": 81.6, "y1": 100.8, "x2": 188.8, "y2": 208.4 },
  "center": { "x": 134.9, "y": 154.6 },
  "quad":   [[…],[…],[…],[…]],

  "difficulty": "easy",
  "random_seed": 955475868,
  "seeds": { "geometry_seed": …, "reference_noise_seed": …, "search_noise_seed": … },

  "architecture": { "P01_word_line_pitch": 11.38, "P02_…": …, "P08_defect_density": 0.0 },
  "P09_scale": 1.049, "P10_rotation_deg": -1.41,
  "footprint_px": 104.9,
  "P17_position_x": 134.9, "P18_position_y": 154.6,
  "reference_imaging": { "blur_sigma": …, "thermal_drift_px": …, "vibration_amp_px": … },
  "search_imaging":    { … },
  "defects": { "broken_line": 0, "missing_contact": 0, … }
}
```

All coordinates are pixels of the final 1000 × 1000 search image, origin
top-left, x right, y down.

### The 20 parameters

P01–P18 are the parameters implied by the five-level randomization hierarchy;
P19 and P20 are the two additional scan-error models.

| | Level 1 — Architecture | | Level 3 — Imaging |
|---|---|---|---|
| P01 | word-line pitch | P09 | scale |
| P02 | bit/word pitch ratio | P10 | rotation |
| P03 | line width fraction | P11 | blur sigma |
| P04 | contact diameter fraction | P12 | brightness |
| P05 | word lines per block | P13 | contrast |
| | **Level 2 — Geometry** | | **Level 4 — SEM effects** |
| P06 | spacing jitter | P14 | edge brightening |
| P07 | lattice phase (x, y) | P15 | spatial intensity variation |
| P08 | defect density | P16 | sensor noise (sigma + dose) |
| | **Level 5 — Placement** | | **Scan errors** |
| P17 | position x | P19 | thermal drift |
| P18 | position y | P20 | vibration |

P11–P16, P19 and P20 are drawn **independently for the reference and the search**
and are stored under `reference_imaging` and `search_imaging` respectively.

### Difficulty distribution

25 % easy / 50 % medium / 25 % hard, shuffled so difficulty does not correlate
with sample index. Harder tiers get more noise, stronger blur, larger rotation
and scale spread, lower contrast, more defects, and heavier drift and vibration.
The tier is recorded per sample and the full range table is in
`metadata/generation_config.json`.

---

## 8. Visualizing samples

```bash
# 6 detail panels + a 25-sample contact sheet
python visualize_dataset.py --dataset dataset --num-panels 6

# one specific sample
python visualize_dataset.py --dataset dataset --image-id sample_000123
```

Figures land in `visualizations/`, deliberately *outside* the dataset directory
so `dataset/` stays exactly the published structure. Each detail panel shows the reference,
the search image, the search image with the ground-truth box and quadrilateral
drawn on, the zoomed ground-truth region, and the sample's full metadata. The
contact sheet is the fastest way to confirm placement really is spread across the
frame and that the difficulty mix looks right.

---

## 9. Reproducibility

One master seed reproduces the dataset byte for byte.

```
master seed
  └── + split offset          train / validation / test draw from disjoint streams
        └── SeedSequence([offset, index])   →  per-sample seed
              └── 6 derived sub-seeds:  params, geometry,
                                        reference noise,  search noise,
                                        reference scan,   search scan
```

Each sample is generated entirely from its own seed and shares no state with any
other, which is both why the run parallelizes cleanly and why the output does not
depend on worker count or completion order. `metadata/generation_config.json`
records the master seed, every parameter range and the exact command to rebuild.

**Leakage.** Splits draw from disjoint seed streams, and every architecture and
geometry parameter is drawn from a continuous distribution, so two samples
sharing a geometry is a measure-zero event. QC check 11 verifies that no sample
seed and no geometry signature appears in more than one split.

---

## 10. Quality control

`generate_dataset.py` runs 12 checks over the finished dataset on disk — the
artefact a consumer actually receives, not in-memory state. Any failure raises
`QCError` naming the affected samples.

1. Exactly the expected number of pairs, per split
2. Every reference file exists
3. Every search file exists
4. Every search image is exactly 1000 × 1000, 8-bit grayscale
5. Every bounding box is valid and its centre lies inside it
6. Every bounding box lies completely inside the search image
7. Every reference is paired with a search image
8. No duplicate filenames **and** no duplicate image content (MD5 over all files)
9. Reference and search noise independently generated — distinct, never-reused
   seeds, plus an empirical high-pass correlation probe
10. Ground truth matches the actual location (see below)
11. Train / validation / test use disjoint seeds, and no geometry is shared
12. No blank or corrupted images

### On check 10

A DRAM array is periodic, so a correlation peak genuinely cannot tell one
lattice repeat from the next. A peak sitting exactly N word-line pitches from
the annotation is matching *ambiguity*, not a coordinate error. Check 10 is
therefore in two parts:

* **`verify_transform_math()`** exercises the corner algebra and the scan-warp
  point mapping on a specimen of smooth **random texture** under deliberately
  heavy drift and vibration. That texture is unique everywhere, so the peak is
  unambiguous and any disagreement is a real coordinate error. This is the part
  that actually proves the annotations are right.
* A **dataset-wide sweep** then only has to catch gross breakage. Offsets that
  resolve to a whole number of lattice steps (≤ 4 steps, sub-2.5 px residual) are
  counted as aliasing; anything else fails, as does a split-wide median above
  3 px, which would indicate a systematic shift.

This periodicity is a real property of the data, not a flaw: naive global
template matching *will* lock onto the wrong repeat sometimes. The mat phases,
the strips and the defects are the cues that disambiguate, which is a large part
of what makes the task interesting.

---

## 11. Files

```
generate_dataset.py      CLI: generate, annotate, run QC, print the summary
visualize_dataset.py     detail panels and contact sheets
requirements.txt         numpy, opencv-python-headless, matplotlib
dram_synth/
  params.py              the 20 parameters, difficulty tiers, seed derivation
  layout.py              procedural DRAM geometry and rasterization
  sem.py                 the imaging chain, including drift and vibration
  sample.py              one sample end to end, with the ground-truth maths
  qc.py                  the 12 checks
```

Generated datasets are excluded from git via `.gitignore` — they are large and
fully reproducible from the seed.
