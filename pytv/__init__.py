# /*-----------------------------------------------------------------------*\
# |                                                                         |
# |                 _____            _______  __      __                    |
# |                |  __ \          |__   __| \ \    / /                    |
# |                | |__) |  _   _     | |     \ \  / /                     |
# |                |  ___/  | | | |    | |      \ \/ /                      |
# |                | |      | |_| |    | |       \  /                       |
# |                |_|       \__/ |    |_|        \/                        |
# |                           __/ |                                         |
# |                          |___/                                          |
# |                                                                         |
# |                                                                         |
# |   Author: E. Boigne                                                     |
# |                                                                         |
# |   Contact: Emeric Boigne                                                |
# |   email: emericboigne@gmail.com                                         |
# |   Department of Mechanical Engineering                                  |
# |   Stanford University                                                   |
# |   488 Escondido Mall, Stanford, CA 94305, USA                           |
# |                                                                         |
# |-------------------------------------------------------------------------|
# |                                                                         |
# |   This file is part of the pyTV package.                                |
# |                                                                         |
# |   License                                                               |
# |                                                                         |
# |   Copyright(C) 2021 E. Boigne                                           |
# |   pyTV is free software: you can redistribute it and/or modify          |
# |   it under the terms of the GNU General Public License as published by  |
# |   the Free Software Foundation, either version 3 of the License, or     |
# |   (at your option) any later version.                                   |
# |                                                                         |
# |   pyTV is distributed in the hope that it will be useful,               |
# |   but WITHOUT ANY WARRANTY; without even the implied warranty of        |
# |   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         |
# |   GNU General Public License for more details.                          |
# |                                                                         |
# |   You should have received a copy of the GNU General Public License     |
# |   along with pyTV. If not, see <http://www.gnu.org/licenses/>.          |
# |                                                                         |
# /*-----------------------------------------------------------------------*/


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
