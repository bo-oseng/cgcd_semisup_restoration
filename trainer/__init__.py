"""
Expose the trainer submodules so callers can write
`from trainer import train_semisup` without worrying about paths.
"""

from importlib import import_module

__all__ = ["train_initialize_old", "train_cgcd_incremental", "train_semisup"]


def __getattr__(name):
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
