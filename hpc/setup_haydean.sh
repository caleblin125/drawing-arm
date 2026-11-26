#!/bin/bash
# One-time setup on the cluster login node: a venv with numpy + scipy (the physics
# test needs nothing else; CadQuery is NOT required).
set -e
python3 -m venv ~/drawarm-venv
~/drawarm-venv/bin/pip install --quiet --upgrade pip
~/drawarm-venv/bin/pip install --quiet numpy scipy matplotlib contourpy pillow
~/drawarm-venv/bin/python -c "import numpy, scipy; print('venv ok', numpy.__version__, scipy.__version__)"
