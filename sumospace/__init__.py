"""
SumoSpace — Locally-first autonomous agent framework.
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("sumospace")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from sumospace.kernel import SumoKernel
from sumospace.settings import SumoSettings
from sumospace.exceptions import (
    SumoSpaceError,
    IngestError,
    ProviderError,
    ProviderNotConfiguredError,
    ConsensusFailedError,
)

__all__ = [
    "SumoKernel",
    "SumoSettings",
    "SumoSpaceError",
    "IngestError",
    "ProviderError",
    "ProviderNotConfiguredError",
    "ConsensusFailedError",
    "__version__",
]
