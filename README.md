---
title: Drift-Sense Synthetic Dataset Generator
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Drift-Sense Synthetic Dataset Generator

Synthetic data generator for the Applied Materials "Drift-Sense" problem
statement (SEMICON India Hackathon 2026 / i4C). No dataset is provided by
the hackathon -- this generates physically-grounded Reference/Search image
pairs with ground truth.

See [`slides/index.html`](slides/index.html) for the full methodology walkthrough &mdash; an HTML slide deck (open directly in a browser, or `python3 -m http.server 8123 --directory slides`). Arrow keys / click edges to navigate, `F` for fullscreen.

## Setup
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Generating a dataset  &mdash;  use this one

`dram_dataset/generate_dataset.py` is **the** dataset generator. It produced
every dataset in this repo. It takes any sample count, splits it 80/10/10,
writes the annotations, and runs 12 quality-control checks over the result.

```
cd dram_dataset
python generate_dataset.py --num-samples 1000 --output ../my_dataset --seed 42
```

| flag | default | meaning |
|---|---|---|
| `--num-samples` | 1000 | total pairs, split 80 / 10 / 10 |
| `--output` | `dataset` | output root directory |
| `--seed` | 42 | master seed; reproduces the dataset |
| `--architecture` | `dram` | device family: `dram` or `finfet`. Same imaging chain, ground truth and draw order for both, so a seed gives identical placement either way |
| `--workers` | auto | processes; auto = `cpu_count-1` capped at 6, `1` = serial |
| `--supersample` | 10 | fine-canvas factor; 6 is ~2x faster, ~1/3 the memory |
| `--skip-qc` | off | skip the 12 checks |

1000 pairs takes about 4 minutes on 6 workers and roughly 900 MB on disk.
Start with `--num-samples 30` to check the generator before a full run.

Output layout:
```
my_dataset/
├── train/       references/  searches/  annotations.json     (800 pairs)
├── validation/  references/  searches/  annotations.json     (100 pairs)
├── test/        references/  searches/  annotations.json     (100 pairs)
└── metadata/    generation_config.json
```

Reference: 100&ndash;256 px square, high magnification.
Search: exactly 1000 x 1000 px, 8-bit grayscale.

**Full documentation** &mdash; the physics, the 20 parameters, the annotation
schema, the QC checks and the reproducibility model &mdash; is in
[`dram_dataset/README.md`](dram_dataset/README.md).

### Visualize samples
```
cd dram_dataset
python visualize_dataset.py --dataset ../my_dataset --num-panels 6
python visualize_dataset.py --dataset ../my_dataset --image-id sample_000123
```
Figures land in `visualizations/`, outside the dataset directory.

### Evaluate a ZNCC baseline on a generated dataset
```
cd dram_dataset
python evaluate_dataset.py --dataset ../my_dataset
```

---

## Curated evaluation suite &mdash; 420 controlled test cases

`dataset_drift/` answers "how accurate overall". It cannot answer "accurate
under *what*", because every pair varies every parameter at once. `eval_suite/`
is the diagnostic complement: 7 stressor axes x 6 levels x 10 replicates, where
each case moves **one** parameter and pins everything else, including all six
random sub-seeds. Axes cover scale, reference resolution ratio, rotation, noise
and dose, blur and scan error, layout periodicity, and combined operational
scenarios.

```
python scripts/make_eval_suite.py --output eval_suite   # render 420 pairs (~3 min)
python scripts/check_eval_suite.py                      # QC gate: PASS/FAIL
python scripts/run_eval_suite.py                        # score v1 and v2 (~11 min)
python scripts/visualize_eval_suite.py --replicate 1    # one figure per axis
```

Results, the rationale for every case, and the honest limitations are in
[`eval_suite/README.md`](eval_suite/README.md); raw numbers in
`reports/eval_suite/`.

---

## Datasets in this repo

| folder | samples | seed | notes |
|---|---|---|---|
| `dataset/` | 1000 | 42 | pre-drift-placement |
| `dataset_2k/` | 1000 | 43 | second roll of the same config |
| `dataset_drift/` | 1000 | 42 | current: drift-realistic placement, 800/100/100 |
| `eval_suite/` | 420 | 11000001&ndash;11000010 | curated diagnostic suite, one axis per case |

All were produced by `dram_dataset/generate_dataset.py` (the suite via
`scripts/make_eval_suite.py`, which drives the same engine) and are reproducible
from the `reproduce_command` recorded in their metadata JSON.

---

## The `src/` pipeline (interactive explorer)

`src/` is a **separate, second** generation pipeline. It supports both DRAM and
FinFET presets and drives the Streamlit explorer and the slide deck. It is not
what produced the datasets above &mdash; do not mix the two.

```
streamlit run app.py                # interactive explorer
```

Its own small-scale CLIs, which write a different layout
(`<split>/reference/`, `<split>/search/`, `manifest.csv`):

```
python generate_dataset.py --num-samples 20 --split train --output-dir ./output --seed 42
python visualize_sample.py --output-dir ./output --split train --id 0
python generate_family_dataset.py    # 1 reference + 5 search variants
python visualize_layers.py           # false-colour per-layer / exploded stack view
```

Run the simple baseline matcher on a pair:
```
python baseline_solution/infer.py --reference output/train/reference/00000.png --search output/train/search/00000.png
```

## Run tests
```
pytest tests/
```
