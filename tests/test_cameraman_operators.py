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
nb_it = 300
regularization = 25
step_size = 5e-3 # If step size is too large, loss function may not decrease at every step

np.random.seed(0)
cameraman_truth = pytv.utils.cameraman() # Open the cameraman's grayscale image
cameraman_noisy = cameraman_truth + noise_level * np.random.rand(*cameraman_truth.shape) # Add noise
cameraman_noisy = np.reshape(cameraman_noisy, (1,1,) + cameraman_noisy.shape)
cameraman_estimate = np.copy(cameraman_noisy)
cameraman_estimate_GD = np.squeeze(np.copy(cameraman_noisy))

loss_fct_GD = np.zeros([nb_it,])
loss_fct = np.zeros([nb_it,])

primal_update = np.zeros_like(cameraman_noisy)
dual_update_fidelity = np.zeros_like(cameraman_noisy)
dual_update_TV = np.zeros_like(cameraman_noisy)

# A simple version of the Chambolle & Pock algorithm for image denoising
# Ref: Chambolle, Antonin, and Thomas Pock. "A first-order primal-dual algorithm for convex problems with applications to imaging." Journal of mathematical imaging and vision 40.1 (2011): 120-145.

sigma_D = 0.5
sigma_A = 1.0
tau = 1 / (8 + 1)

for it in range(nb_it):

    # Dual update
    dual_update_fidelity = (dual_update_fidelity + sigma_A * (cameraman_estimate - cameraman_noisy))/(1.0+sigma_A)
    D_x = pytv.tv_operators_GPU.D_hybrid(cameraman_estimate)
    prox_argument = dual_update_TV + sigma_D * D_x
    dual_update_TV = prox_argument / np.maximum(1.0, np.sqrt(np.sum(prox_argument**2, axis = 1)) / regularization)

    # Primal update
    cameraman_estimate = cameraman_estimate - tau * dual_update_fidelity - tau * pytv.tv_operators_GPU.D_T_hybrid(dual_update_TV)

    # Loss function update
    loss_fct[it] = 0.5 * np.sum(np.square(cameraman_estimate - cameraman_noisy)) + regularization * pytv.tv_operators_GPU.compute_L21_norm(D_x)

    # Comparisons with gradient descent
    tv, G = pytv.tv_GPU.tv_hybrid(cameraman_estimate_GD)
    cameraman_estimate_GD += - step_size * ((cameraman_estimate_GD - np.squeeze(cameraman_noisy)) + regularization * G)
    loss_fct_GD[it] = 0.5 * np.sum(np.square(cameraman_estimate_GD - np.squeeze(cameraman_noisy))) + regularization * tv


plt.figure(1, figsize=[9.5, 3], dpi = 150)
plt.subplot(1,3,1, title='Truth (no noise)')
plt.imshow(cameraman_truth, cmap = plt.get_cmap('gray'))
plt.axis('off')
plt.subplot(1,3,2, title='Noisy input')
plt.axis('off')
plt.imshow(np.squeeze(cameraman_noisy), cmap = plt.get_cmap('gray'))
plt.subplot(1,3,3, title='Algorithm output')
plt.imshow(np.squeeze(cameraman_estimate), cmap = plt.get_cmap('gray'))
plt.axis('off')
plt.tight_layout(pad=0.5)

loss_fct_min = 37627498.5 # Achieved with 1000 iterations
plt.figure(2, figsize=[3, 2], dpi = 75)
plt.loglog(loss_fct-loss_fct_min, label = 'CP')
plt.loglog(loss_fct_GD-loss_fct_min, label = 'GD')
plt.xlabel('Iteration')
plt.ylabel('Loss function: f-f^*')
plt.tight_layout(pad=0.5)
plt.legend()
plt.show()



