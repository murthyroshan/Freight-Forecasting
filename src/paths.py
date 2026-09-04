"""Filesystem layout for the project.

Every path is resolved from the location of this file, not from the
current working directory. That means a script behaves identically
whether you run

    python src/train_model.py
    python -m src.train_model
    python train_model.py        (from inside src/)

or launch it from an editor with some third working directory. Relative
paths like 'data/raw' silently produce an empty dataset when the cwd is
wrong, and an empty dataset is exactly the failure mode this repo is
built to avoid.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA = os.path.join(ROOT, 'data')
RAW = os.path.join(DATA, 'raw')
PROCESSED = os.path.join(DATA, 'processed')
MODELS = os.path.join(ROOT, 'models')
TEMPLATES = os.path.join(ROOT, 'templates')


def bootstrap():
    """Put the repo root on sys.path so `from src import ...` resolves
    even when a file is executed directly rather than as a module."""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)


def ensure_dirs():
    for d in (RAW, PROCESSED, MODELS):
        os.makedirs(d, exist_ok=True)
