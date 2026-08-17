"""Marks `dram_dataset` as a package so the canonical engine can be imported
from the repository root as `dram_dataset.dram_synth`.

The scripts inside this directory still put their own directory on `sys.path`
and `import dram_synth` directly, which continues to work unchanged; this only
adds the qualified path so code living at the repo root (the Streamlit app, the
slide asset builder) can share the same engine instead of duplicating it.
"""
