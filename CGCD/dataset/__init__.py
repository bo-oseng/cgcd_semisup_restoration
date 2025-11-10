from .degradations import DegradationDataset

from . import utils
from .base import BaseDataset


_type = {
    "deg": DegradationDataset,
}


def load(name, root, mode, transform=None):
    return _type[name](root=root, mode=mode, transform=transform)
