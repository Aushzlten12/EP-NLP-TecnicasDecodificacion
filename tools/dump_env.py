# tools/dump_env.py
import platform, sys

print("PYTHON", sys.version.replace("\n", " "))
try:
    import numpy as np

    print("NUMPY", np.__version__)
except Exception:
    print("NUMPY none")
try:
    import torch

    print("TORCH", torch.__version__)
except Exception:
    print("TORCH none")
print("PLATFORM", platform.platform())
