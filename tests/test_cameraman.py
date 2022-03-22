# /*-----------------------------------------------------------------------*\
# |                                                                         |
# |      _____            _______  __      __           _  _   _____        |
# |     |  __ \          |__   __| \ \    / /          | || | |  __ \       |
# |     | |__) |  _   _     | |     \ \  / /   ______  | || |_| |  | |      |
# |     |  ___/  | | | |    | |      \ \/ /   |______| |__   _| |  | |      |
# |     | |      | |_| |    | |       \  /                | | | |__| |      |
# |     |_|       \__, |    |_|        \/                 |_| |_____/       |
# |                __/ |                                                    |
# |               |___/                                                     |
# |                                                                         |
# |                                                                         |
# |   Author: Emeric Boigné                                                 |
# |                                                                         |
# |   Contact: Emeric Boigné                                                |
# |   email: emericboigne@gmail.com                                         |
# |   Department of Mechanical Engineering                                  |
# |   Stanford University                                                   |
# |   488 Escondido Mall, Stanford, CA 94305, USA                           |
# |                                                                         |
# |-------------------------------------------------------------------------|
# |                                                                         |
# |   This file is part of the PyTV-4D package.                             |
# |                                                                         |
# |   License                                                               |
# |                                                                         |
# |   Copyright(C) 2021 E. Boigné                                           |
# |   PyTV-4D is free software: you can redistribute it and/or modify       |
# |   it under the terms of the GNU General Public License as published by  |
# |   the Free Software Foundation, either version 3 of the License, or     |
# |   (at your option) any later version.                                   |
# |                                                                         |
# |   PyTV-4D is distributed in the hope that it will be useful,            |
# |   but WITHOUT ANY WARRANTY; without even the implied warranty of        |
# |   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         |
# |   GNU General Public License for more details.                          |
# |                                                                         |
# |   You should have received a copy of the GNU General Public License     |
# |   along with PyTV-4D. If not, see <http://www.gnu.org/licenses/>.       |
# |                                                                         |
# /*-----------------------------------------------------------------------*/


import pytv
import numpy as np
import matplotlib.pyplot as plt

noise_level = 100
nb_it = 150
regularization = 25
step_size = 5e-3 # If step size is too large, loss function may not decrease at every step

cameraman_truth = pytv.utils.cameraman() # Open the cameraman's grayscale image
cameraman_noisy = cameraman_truth + noise_level * np.random.rand(*cameraman_truth.shape) # Add noise
cameraman_estimate = np.copy(cameraman_noisy)

loss_fct = np.zeros([nb_it,])
for it in range(nb_it): # A simple sub-gradient descent algorithm for image denoising
    try:
        tv, G = pytv.tv_CPU.tv_hybrid(cameraman_estimate)
    except:
        tv, G = pytv.tv_GPU.tv_hybrid(cameraman_estimate)
    cameraman_estimate += - step_size * ((cameraman_estimate - cameraman_noisy) + regularization * G)
    loss_fct[it] = 0.5 * np.sum(np.square(cameraman_estimate - cameraman_noisy)) + regularization * tv

plt.figure(1, figsize=[9.5, 3], dpi = 150)
plt.subplot(1,3,1, title='Truth (no noise)')
plt.imshow(cameraman_truth, cmap = plt.get_cmap('gray'))
plt.axis('off')
plt.subplot(1,3,2, title='Noisy input')
plt.axis('off')
plt.imshow(cameraman_noisy, cmap = plt.get_cmap('gray'))
plt.subplot(1,3,3, title='Algorithm output')
plt.imshow(cameraman_estimate, cmap = plt.get_cmap('gray'))
plt.axis('off')
plt.tight_layout(pad=0.5)

plt.figure(2, figsize=[3, 2], dpi = 75)
plt.plot(loss_fct)
plt.xlabel('Iteration')
plt.ylabel('Loss function')
plt.tight_layout(pad=0.5)
plt.show()
