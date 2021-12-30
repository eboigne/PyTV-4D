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


import pytv
import numpy as np

Nz, N = 20, 1000 # 3D Image dimensions
np.random.seed(0)
img = np.random.rand(Nz, N, N)

tv1, G1 = pytv.tv_CPU.tv_hybrid(img)
tv2, G2 = pytv.tv_GPU.tv_hybrid(img)

print('TV value from CPU: '+str(tv1))
print('TV value from GPU: '+str(tv2))
print('Sub-gradients from CPU and GPU are equal: '+str(np.prod(np.abs(G1-G2)<1e-5)>0))
