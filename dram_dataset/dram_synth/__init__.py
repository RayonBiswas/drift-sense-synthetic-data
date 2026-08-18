"""Procedural DRAM-style SEM synthetic dataset generator for reference-to-search
image localization.

Modules
-------
params  : the 20 randomized generation parameters (P01-P20) and difficulty tiers
layout  : procedural DRAM geometry (word lines, bit lines, contacts, mats, defects)
sem     : SEM imaging chain (edge brightening, thermal drift, vibration, blur,
          shading, contrast/brightness, independent sensor noise)
sample  : end-to-end generation of one sample, with analytic ground truth
qc      : the 12 quality-control checks
"""

__version__ = "1.0.0"
