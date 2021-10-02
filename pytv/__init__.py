from . import tv_2d_CPU
from . import tv_CPU
from . import tv_operators_CPU

try:
    import torch
    from . import tv_2d_GPU
    from . import tv_GPU
    from . import tv_operators_GPU
except (ImportError, ModuleNotFoundError) as error:
    print('PyTorch not properly imported, imported CPU routines only:')
    print(error)

from . import utils
from . import tests

from .utils import *
from .tests import *

# Keep at end
__all__ = ['tv_CPU', 'tv_GPU', 'tv_operators_CPU', 'tv_operators_GPU', 'tv_2d_CPU', 'tv_2d_GPU', 'utils', 'tests']
