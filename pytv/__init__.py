__all__ = ['tv', 'tv_pyTorch', 'tv_operators', 'tv_operators_pyTorch', 'tv_2d', 'tv_pyTorch_2d', 'utils']

from . import tv_2d
from . import tv
from . import tv_operators
from . import utils

try:
    import torch
    from . import tv_pyTorch_2d
    from . import tv_pyTorch
    from . import tv_operators_pyTorch
except (ImportError, ModuleNotFoundError) as error:
    print('PyTorch not properly imported, imported CPU routines only:')
    print(error)
    # raise
