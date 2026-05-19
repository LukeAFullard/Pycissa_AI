# pycissa/__init__.py

# Package metadata
__version__ = "0.1.0"      # keep in sync with pyproject.toml
__author__  = "Luke A. Fullard"
__license__ = "MIT"

# Expose only the top‐level API
from .processing.cissa.cissa import Cissa
from .processing.mcissa.mcissa import MCissa

__all__ = [
    "Cissa",
    "__version__",
]
