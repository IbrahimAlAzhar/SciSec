#!/usr/bin/env python3
"""
run_with_paths.py — run any script of this release with every sub-folder on the import
path (modules import their neighbours by plain name).

    python run_with_paths.py <sub-folder>/<script>.py [script arguments]

Paths to data, models and outputs are given on the command line or as environment
variables; edit the defaults in each script for your setup.
"""
import os, runpy, sys

here = os.path.dirname(os.path.abspath(__file__))
for d in sorted(os.listdir(here)):
    if os.path.isdir(os.path.join(here, d)) and not d.startswith((".", "_")):
        sys.path.insert(0, os.path.join(here, d))
if len(sys.argv) < 2:
    sys.exit(__doc__)
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
