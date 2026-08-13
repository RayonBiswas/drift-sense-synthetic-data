# Drift-Sense Synthetic Dataset Generator — Workspace Analysis

## 1. Project Purpose

This is a **synthetic SEM image dataset generator** for the Applied Materials "Drift-Sense" problem statement (SEMICON India Hackathon 2026 / i4C). The project creates physically-grounded semiconductor reference/search image pairs with ground-truth annotations for machine learning.

### Key Objectives
- Generate 1,000 high-quality DRAM-pattern SEM image pairs (with optional FinFET patterns)
- Model realistic semiconductor physics: structural geometry, defects, and SEM acquisition artifacts
- Provide precise ground-truth annotations (bounding boxes, centers, rotation, scale)
- Serve as a benchmark dataset for learning-based and classical image matching algorithms

### Image Specification
- **Reference image**: 1000×1000 px @ 1 nm/px (1 μm FOV)  
- **Search image**: 1000×1000 px @ 10 nm/px (10 μm FOV)
- **Dataset split**: 800 training, 100 validation, 100 test (1000 total)
- **Physical effects modeled**: beam spot, dose, charging streaks, raster drift, noise, distortion, vignetting

### Architecture Types
1. **DRAM**: Folded-bitline 6F² cell array (horizontal word-lines, vertical bit-lines, periodic contacts)
2. **FinFET**: Fin/gate stack structure (scalable to 7 nm, 10 nm, 14 nm nodes)
3. **Zoned layouts**: Multi-region mats separated by routing strips (realistic die composition)

---

## 2. Code Summary

### **Core Entry Points**

| File | Purpose |
|------|---------|
| [generate_dataset.py](generate_dataset.py) | CLI to generate a dataset split (train/validation/test) with configurable physics parameters |
| [generate_synthetic_dataset.py](generate_synthetic_dataset.py) | Older standalone generator; generates ~30–1000 samples with JSON metadata |
| [generate_family_dataset.py](generate_family_dataset.py) | Creates sample families: one reference + 5 search variants at different acquisition conditions |
| [app.py](app.py) | Streamlit interactive explorer; UI shell over `src/pipeline.py` |

### **src/ Module (Core Generation Pipeline)**

| File | Purpose |
|------|---------|
| [src/pipeline.py](src/pipeline.py#L1-L150) | Orchestrates sample generation: fine canvas → reference crop → search image with ground-truth |
| [src/presets.py](src/presets.py) | Preset architecture specs (DRAM_1X, DRAM_DENSE, FINFET_10NM, etc.) in nanometers |
| [src/sem_imaging.py](src/sem_imaging.py) | SEM physics: PSF blur, vignetting, gamma, barrel distortion, charging streaks, shot/detector noise, raster drift |
| [src/structural_defects.py](src/structural_defects.py) | Pattern imperfections: line-width jitter, gap collapse, broken lines, contact missing |
| [src/patterns/dram.py](src/patterns/dram.py) | Render DRAM cell arrays (word-lines, bit-lines, contacts); vectorized 1D masks for speed |
| [src/patterns/finfet.py](src/patterns/finfet.py) | Render FinFET fin/gate stacks |
| [src/patterns/zones.py](src/patterns/zones.py) | Multi-mat zoning with boundary strip composition (realistic die layout) |

### **dram_dataset/ Module (Alternative DRAM-Only Generator)**

A parallel, self-contained generator tuned for DRAM with difficulty tiers:

| File | Purpose |
|------|---------|
| [dram_dataset/generate_dataset.py](dram_dataset/generate_dataset.py) | CLI for DRAM-specific dataset with worker parallelization |
| [dram_dataset/evaluate_dataset.py](dram_dataset/evaluate_dataset.py) | ZNCC baseline evaluation on a saved dataset (precision-recall, AP, lattice-explained errors) |
| [dram_dataset/visualize_dataset.py](dram_dataset/visualize_dataset.py) | Contact sheets and detailed layer visualizations |
| [dram_dataset/dataset_generator.py](dram_dataset/dataset_generator.py) | Adapter exposing the `dram_synth` pipeline; generates on-demand without saved dataset |
| [dram_dataset/dram_synth/sample.py](dram_dataset/dram_synth/sample.py) | End-to-end sample rendering: fine canvas → reference window → both image captures |
| [dram_dataset/dram_synth/layout.py](dram_dataset/dram_synth/layout.py) | Zoned canvas rendering (mats + strips) |
| [dram_dataset/dram_synth/params.py](dram_dataset/dram_synth/params.py) | Difficulty tiers, seed management, parameter sampling |
| [dram_dataset/dram_synth/sem.py](dram_dataset/dram_synth/sem.py) | Image capture, SEM degradation, thermal/vibrational drift simulation |
| [dram_dataset/dram_synth/qc.py](dram_dataset/dram_synth/qc.py) | Quality-control checks: pattern stats, layer validation, GT correlation verification |
| [dram_dataset/dram_synth/random.py](dram_dataset/dram_synth/random.py) | Artifact functions (noise, drift, etc.) |

### **baseline_solution/ (Baseline Matcher)**

| File | Purpose |
|------|---------|
| [baseline_solution/infer.py](baseline_solution/infer.py) | Simple inference: load reference, search; run ZNCC at multiple scales; report match |
| [baseline_solution/zncc.py](baseline_solution/zncc.py) | Explicit ZNCC (zero-mean normalized cross-correlation) matching; multi-scale sweep |
| [baseline_solution/evaluate.py](baseline_solution/evaluate.py) | Comprehensive evaluation: generate samples at 4 noise levels, sweep ZNCC score threshold, report precision-recall curves & AP |

### **Utilities & Visualizations**

| File | Purpose |
|------|---------|
| [visualize_sample.py](visualize_sample.py) | Draw ground-truth box on search, show reference thumbnail side-by-side |
| [visualize_synthetic_dataset.py](visualize_synthetic_dataset.py) | Contact sheets & detailed sample inspections from saved dataset |
| [visualize_layers.py](visualize_layers.py) | False-color per-layer rendering + exploded stack view for teaching/QA |
| [slides/make_assets.py](slides/make_assets.py) | Generate slide assets (example images, comparison grids) |

### **Tests**

| File | Purpose |
|------|---------|
| [tests/test_pipeline.py](tests/test_pipeline.py) | Parametrized tests: verify ground-truth patch matches reference; check image shapes & bounds |

---

## 3. Code Repetitions

### **A. Image I/O Patterns**

**Pattern**: `cv2.imread()` + `cv2.imwrite()` scattered across 11 files

**Files and occurrences**:
- [baseline_solution/infer.py](baseline_solution/infer.py) — Lines 25–26: Load reference/search with `cv2.IMREAD_GRAYSCALE`
- [baseline_solution/zncc.py](baseline_solution/zncc.py) — Lines 71–72: Same pattern
- [dram_dataset/dram_synth/qc.py](dram_dataset/dram_synth/qc.py) — Lines 105–106 (IMREAD_UNCHANGED), 192–193, 416–417 (IMREAD_GRAYSCALE)
- [dram_dataset/evaluate_dataset.py](dram_dataset/evaluate_dataset.py) — Lines 140–141
- [dram_dataset/visualize_dataset.py](dram_dataset/visualize_dataset.py) — Lines 62–63
- [generate_dataset.py](generate_dataset.py) — Lines 104–105: Write without flags (compression)
- [generate_family_dataset.py](generate_family_dataset.py) — Lines 56, 61: Write with `cv2.imwrite()`
- [generate_synthetic_dataset.py](generate_synthetic_dataset.py) — Lines 606–607 (write), 693, 711 (read)
- [visualize_sample.py](visualize_sample.py) — Lines 37–38 (read), 58 (write)
- [visualize_synthetic_dataset.py](visualize_synthetic_dataset.py) — Lines 75, 135–136

**Issue**: No centralized utility; each module re-implements read/write with slightly different flags and error handling.

**Recommendation**: Extract to `src/io_utils.py`:
```python
def load_image_grayscale(path: str) -> np.ndarray
def save_image(path: str, img: np.ndarray, quality: int = 3)
```

---

### **B. Random Seed Initialization**

**Pattern**: `np.random.default_rng(seed)` repeated 40 times across 17 files

**Files**:
- [generate_dataset.py](generate_dataset.py#L51): `rng = np.random.default_rng(args.seed)`
- [generate_family_dataset.py](generate_family_dataset.py#L32): Same
- [generate_synthetic_dataset.py](generate_synthetic_dataset.py#L485)
- [app.py](app.py#L160), [app.py](app.py#L286)
- [baseline_solution/evaluate.py](baseline_solution/evaluate.py#L72)
- [dram_dataset/dataset_generator.py](dram_dataset/dataset_generator.py#L58): With seed hashing
- [dram_dataset/dram_synth/params.py](dram_dataset/dram_synth/params.py) — Lines 194, 258
- [dram_dataset/dram_synth/qc.py](dram_dataset/dram_synth/qc.py#L337)
- [dram_dataset/dram_synth/sem.py](dram_dataset/dram_synth/sem.py) — Lines 97, 242
- [dram_dataset/evaluate.py](dram_dataset/evaluate.py) — Lines 37, 46
- [src/pipeline.py](src/pipeline.py) — Lines 251, 257, 283
- [tests/test_pipeline.py](tests/test_pipeline.py) — Lines 29, 50
- [visualize_layers.py](visualize_layers.py#L111)
- [visualize_synthetic_dataset.py](visualize_synthetic_dataset.py) — Lines 52, 340
- [slides/make_assets.py](slides/make_assets.py) — Multiple instances

**Issue**: No standard seed-to-RNG factory; various approaches to seed hashing (e.g., `(seed * 2654435761) % (2**32)` vs. `seed * 3 + 7`).

**Recommendation**: Create `src/random_utils.py`:
```python
def make_rng(seed: int, namespace: str = "") -> np.random.Generator
    """Hash seed with namespace to avoid collisions across modules."""
```

---

### **C. Directory Creation**

**Pattern**: `os.makedirs(path, exist_ok=True)` in 5 files

**Occurrences**:
- [generate_dataset.py](generate_dataset.py) — Lines 78–79: Create ref_dir, search_dir
- [generate_family_dataset.py](generate_family_dataset.py) — Lines 35, 51
- [baseline_solution/evaluate.py](baseline_solution/evaluate.py#L105)
- [visualize_layers.py](visualize_layers.py#L106)
- [slides/make_assets.py](slides/make_assets.py#L27)

**Issue**: Repeated boilerplate; no error handling for permission issues.

**Recommendation**: Extract to `src/fs_utils.py`:
```python
def ensure_dir(path: str) -> Path
```

---

### **D. CLI Argument Parsing**

**Pattern**: 7 files define `parse_args()` with similar structure and overlapping parameters

**Files**:
- [generate_dataset.py](generate_dataset.py#L20): ~20 generation params
- [generate_family_dataset.py](generate_family_dataset.py#L21): Architecture + output config
- [visualize_sample.py](visualize_sample.py#L17): output-dir, split, id
- [visualize_layers.py](visualize_layers.py#L51): architecture, size, seed, collapse-threshold
- [dram_dataset/generate_dataset.py](dram_dataset/generate_dataset.py#L38): num-samples, output, seed, workers, supersample
- [dram_dataset/evaluate_dataset.py](dram_dataset/evaluate_dataset.py#L64): dataset, splits, tolerance, workers, sizes, angles
- [dram_dataset/visualize_dataset.py](dram_dataset/visualize_dataset.py#L38): dataset, output, seed

**Issue**: Common params (seed, output, architecture) defined independently; no shared argument groups.

**Recommendation**: Create `src/cli_utils.py`:
```python
def add_seed_args(parser)
def add_output_args(parser)
def add_architecture_args(parser)
```

---

### **E. Ground-Truth and Bounding-Box Extraction**

**Pattern**: Similar box unpacking in [dram_dataset/evaluate_dataset.py](dram_dataset/evaluate_dataset.py), [visualize_sample.py](visualize_sample.py), [dram_dataset/visualize_dataset.py](dram_dataset/visualize_dataset.py)

**Example** (3 similar unpacking patterns):
```python
# Pattern 1: visualize_sample.py line 42
x0, y0, w, h = (float(row["gt_box_x"]), float(row["gt_box_y"]), ...)

# Pattern 2: dram_dataset/evaluate_dataset.py line 145
x0, y0, x1, y1 = (float(r["x1"]), float(r["y1"]), ...)

# Pattern 3: dram_dataset/visualize_dataset.py line 98
x1, y1, x2, y2 = (float(rec["x1"]), float(rec["y1"]), ...)
```

**Issue**: Different CSV column names / annotation formats across generators.

**Recommendation**: Standardize on a single `GroundTruth` dataclass and utility to unpack from dict.

---

### **F. CSV Manifest Writing**

**Pattern**: Similar CSV writer setup in [generate_dataset.py](generate_dataset.py), [generate_family_dataset.py](generate_family_dataset.py), [dram_dataset/generate_dataset.py](dram_dataset/generate_dataset.py)

**Example**:
- [generate_family_dataset.py](generate_family_dataset.py#L35-L42): Open CSV, write headers, write rows in loop
- [generate_dataset.py](generate_dataset.py) — Comparable structure (not shown in excerpt, but implied)

**Issue**: Redundant CSV setup logic; different column orderings.

**Recommendation**: Extract to `src/manifest_utils.py` with a standard `write_manifest()` function.

---

## 4. File Importance Ranking

### **Tier 1: Core Pipeline (Essential)**

1. **[src/pipeline.py](src/pipeline.py)** — Central orchestrator; all CLI generators depend on this. Defines `GenerationParams`, `generate_sample()`, `generate_sample_family()`.
2. **[src/presets.py](src/presets.py)** — Architecture definitions (DRAM_1X, FINFET_10NM, etc.). Required by all generators.
3. **[src/patterns/dram.py](src/patterns/dram.py)** — DRAM rendering; heavily used for test/demo samples.
4. **[src/patterns/finfet.py](src/patterns/finfet.py)** — FinFET rendering; alternative architecture.
5. **[src/sem_imaging.py](src/sem_imaging.py)** — Imaging physics (blur, noise, drift); applied to every image.

### **Tier 2: Generation / Evaluation Scripts (High Use)**

6. **[generate_dataset.py](generate_dataset.py)** — Main CLI for dataset generation; likely most-run script.
7. **[baseline_solution/evaluate.py](baseline_solution/evaluate.py)** — Comprehensive baseline evaluation; reports dataset difficulty.
8. **[generate_family_dataset.py](generate_family_dataset.py)** — Variant generation; demonstrates robustness.
9. **[dram_dataset/generate_dataset.py](dram_dataset/generate_dataset.py)** — Parallel DRAM-only generator; alternative entry point.

### **Tier 3: Utilities & QA**

10. **[src/patterns/zones.py](src/patterns/zones.py)** — Zoned/multi-mat layouts; used in advanced samples.
11. **[src/structural_defects.py](src/structural_defects.py)** — Pattern imperfections; adds realism.
12. **[dram_dataset/dram_synth/qc.py](dram_dataset/dram_synth/qc.py)** — Quality checks; validates dataset consistency.
13. **[tests/test_pipeline.py](tests/test_pipeline.py)** — Regression tests; ensures generator correctness.

### **Tier 4: Visualization & Exploration**

14. **[app.py](app.py)** — Streamlit interactive explorer; educational/debugging tool.
15. **[visualize_sample.py](visualize_sample.py)** — Quick sample inspection.
16. **[visualize_layers.py](visualize_layers.py)** — Layer visualization; teaching aid.
17. **[visualize_synthetic_dataset.py](visualize_synthetic_dataset.py)** — Dataset-level visualization.

### **Tier 5: Supporting / Experimental**

18. **[baseline_solution/infer.py](baseline_solution/infer.py)** — Simple inference wrapper.
19. **[baseline_solution/zncc.py](baseline_solution/zncc.py)** — ZNCC matcher (used by evaluation).
20. **[generate_synthetic_dataset.py](generate_synthetic_dataset.py)** — Older, standalone generator; largely superseded by `generate_dataset.py`.
21. **[dram_dataset/dataset_generator.py](dram_dataset/dataset_generator.py)** — Adapter; used internally by DRAM harness.
22. **[dram_dataset/dram_synth/\*.py](dram_dataset/dram_synth/)** — Internal to DRAM generator; not typically called directly.
23. **[slides/make_assets.py](slides/make_assets.py)** — One-off slide generation; not core to data generation.

### **Rationale**

- **Tier 1** is the immutable foundation; changes here ripple everywhere.
- **Tier 2** are high-use entry points; CLI stability depends on these.
- **Tier 3** provide necessary features (validation, defects, zones).
- **Tier 4** are "nice to have" for exploration but not required for dataset generation.
- **Tier 5** are auxiliary, experimental, or superseded code paths.

---

## 5. Progress Status

### **Completed (Production-Ready)**

✅ **Core generation pipeline** ([src/pipeline.py](src/pipeline.py), [src/presets.py](src/presets.py))
- Well-structured, documented, tested
- Supports DRAM and FinFET architectures
- Realistic physics: beam blur, dose, drift, noise, distortion

✅ **DRAM pattern rendering** ([src/patterns/dram.py](src/patterns/dram.py))
- Vectorized for speed; handles 10,000×10,000 px fine canvases efficiently
- Per-instance jitter; line-width variations; contact checklists

✅ **SEM imaging physics** ([src/sem_imaging.py](src/sem_imaging.py))
- Comprehensive artifact suite: vignetting, gamma, barrel distortion, charging streaks, raster drift
- Separate reference/search degradation

✅ **CLI generation scripts** ([generate_dataset.py](generate_dataset.py), [generate_family_dataset.py](generate_family_dataset.py))
- Configurable; supports multiple architectures
- CSV manifest with ground truth
- Parallelizable (worker pool ready)

✅ **Baseline matcher & evaluation** ([baseline_solution/](baseline_solution/), [baseline_solution/evaluate.py](baseline_solution/evaluate.py))
- ZNCC template matching at multiple scales/rotations
- Precision-recall curves; AP metric
- Comprehensive noise-level testing

✅ **Testing** ([tests/test_pipeline.py](tests/test_pipeline.py))
- Ground-truth validation: confirms crop at GT box matches reference
- Image shape/bounds checks

✅ **Documentation**
- README.md: Quick-start, setup, usage
- SYNTHETIC_DATASET_README.md: Architecture, degradation levels, methodology
- Code docstrings throughout

---

### **Mature (Stable, Documented)**

⚡ **FinFET pattern rendering** ([src/patterns/finfet.py](src/patterns/finfet.py))
- Implemented; used in baseline evaluation
- Less frequently tested than DRAM but functional

⚡ **Zoned/multi-mat layouts** ([src/patterns/zones.py](src/patterns/zones.py))
- Realistic die structure (array mats + routing strips)
- Boundary-biased crop sampling (optional)

⚡ **Quality-control checks** ([dram_dataset/dram_synth/qc.py](dram_dataset/dram_synth/qc.py))
- Validates generator outputs; detects anomalies
- Correlates ground-truth box with actual image content

⚡ **Streamlit explorer** ([app.py](app.py))
- Interactive parameter sweep
- Real-time visualization
- Thin wrapper over [src/pipeline.py](src/pipeline.py) — no separate re-implementation

---

### **In Progress / Partially Implemented**

🔄 **Alternative DRAM generator** ([dram_dataset/](dram_dataset/))
- Self-contained, parallel implementation
- Difficulty tiers (easy/medium/hard/severe)
- Independent from [src/pipeline.py](src/pipeline.py); some code duplication
- **Status**: Functional but parallel to main pipeline (not merged)

🔄 **Structural defects** ([src/structural_defects.py](src/structural_defects.py))
- Gap collapse, line breaks, contact missing
- Per-sample jitter
- **Status**: Implemented but may need expanded defect taxonomy for realism

---

### **Areas of Potential Future Work**

❓ **Code consolidation**
- Merge `src/` and `dram_dataset/dram_synth/` generators (currently parallel code paths)
- Standardize CLI argument parsing and manifest formats
- Extract shared I/O and utility functions

❓ **Advanced defects**
- Overlay/lithography pattern collapse
- Line-end shortening, via bridging
- More realistic etch bias / OPC effects

❓ **Multi-architecture datasets**
- Mixed DRAM/FinFET splits in a single run
- Cross-architecture evaluation

❓ **Scaling & performance**
- GPU-accelerated rendering (high-res, high-throughput)
- Distributed generation (cloud job submission)

❓ **Metadata enrichment**
- Layer-by-layer annotations (for interpretability)
- Defect occurrence tags (which gaps collapsed? which contacts missing?)
- Imaging condition metadata (dose, drift amount, etc.)

---

## Summary Table

| Aspect | Status |
|--------|--------|
| **DRAM pattern generation** | ✅ Complete, production |
| **FinFET pattern generation** | ⚡ Implemented, less tested |
| **Zoning/mat composition** | ⚡ Implemented, optional |
| **SEM physics (beam, noise, drift, distortion)** | ✅ Complete |
| **CLI dataset generation** | ✅ Complete |
| **Baseline evaluation** | ✅ Complete |
| **Unit tests** | ✅ Core tests present |
| **Documentation** | ✅ README + docstrings |
| **Code quality** | 🔄 Good structure; some duplication in dram_dataset/ |
| **Parallelization** | ⚡ Partial (workers in dram_dataset only) |
| **Difficulty tiers** | ⚡ In dram_dataset; not in main pipeline |

