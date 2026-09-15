"""Physim's native taskset and prepared-world runtime."""

from importlib import import_module

__version__ = "0.12.2"
__all__ = ["PhysimTaskset", "Bundle", "BundleError", "ExperimentService", "OracleRunner"]
_PUBLIC = {
    "PhysimTaskset": ("physim.taskset", "R6Taskset"),
    "Bundle": ("physim.bundles", "Bundle"),
    "BundleError": ("physim.bundles", "BundleError"),
    "ExperimentService": ("physim.blobround6_explore", "ExperimentService"),
    "OracleRunner": ("physim.blobround6", "OracleRunner"),
}


def __getattr__(name):
    if name in _PUBLIC:
        module, attribute = _PUBLIC[name]
        return getattr(import_module(module), attribute)
    raise AttributeError(name)
