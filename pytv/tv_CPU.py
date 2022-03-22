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


import numpy as np
import pytv

def tv_hybrid(img, mask = [], reg_z_over_reg = 1.0, reg_time = 0.0, mask_static = False, factor_reg_static = 0, match_2D_form = False):
    '''
    Calculates the total variation and a subgradient of the input image img using the hybrid gradient discretization

    Parameters
    ----------
    img : np.ndarray
        The array of the input image data of dimensions Nz x M x N x N.
    mask : np.ndarray
        A mask used to specify regions of the image ot skip for the TV calculation.
    reg_z_over_reg : float
        The ratio of the regularization parameter in the z direction, versus the x-y plane.

    Returns
    -------
    tv : float
        The value of the total variation.
    G : np.ndarray
        A sub-gradient array of the total variation term of same dimensions as the input image img.
    '''

    if mask != []:
        img[~mask] = 0
    Nz = img.shape[0]
    M = img.shape[1]

    D_img = pytv.tv_operators_CPU.D_hybrid(img, reg_z_over_reg = reg_z_over_reg, reg_time = reg_time, mask_static = mask_static, factor_reg_static = factor_reg_static)
    tv, grad_norms = pytv.tv_operators_CPU.compute_L21_norm(D_img, return_array=True)

    # When non-differentiable, set to 0.
    grad_norms[grad_norms == 0] = np.inf
    G = np.zeros_like(img)

    # Upwind terms along rows & columns
    G[:, :, :, :] += -(D_img[:,0,:,:,:]+D_img[:,1,:,:,:]) / grad_norms[:, :, :, :]
    G[:, :, 1:, :] += D_img[:,0,:,:-1,:] / grad_norms[:, :, :-1, :]
    G[:, :, :, 1:] += D_img[:,1,:,:,:-1] / grad_norms[:, :, :, :-1]

    # Downwind terms along rows & columns
    G[:, :, :, :] += (D_img[:,2,:,:,:]+D_img[:,3,:,:,:]) / grad_norms[:, :, :, :]
    G[:, :, :-1, :] += -D_img[:,2,:,1:,:] / grad_norms[:, :, 1:, :]
    G[:, :, :, :-1] += -D_img[:,3,:,:,1:] / grad_norms[:, :, :, 1:]

    i_d = 4
    if Nz > 1 and reg_z_over_reg > 0:
        # Upwind terms along slices
        G[:, :, :, :] += - D_img[:,i_d,:,:,:] / grad_norms[:, :, :, :]
        G[1:, :, :, :] += D_img[:-1,i_d,:,:,:] / grad_norms[:-1, :, :, :]
        i_d += 1

        # Downwind terms along slices
        G[:, :, :, :] += D_img[:,i_d,:,:,:] / grad_norms[:, :, :, :]
        G[:-1, :, :, :] += -D_img[1:,i_d,:,:,:] / grad_norms[1:, :, :, :]
        i_d += 1

    if reg_time > 0 and M > 1:
        # Upwind terms along time
        G[:, :, :, :] += - D_img[:,i_d,:,:,:] / grad_norms[:, :, :, :]
        G[:, 1:, :, :] += D_img[:,i_d,:-1,:,:] / grad_norms[:, :-1, :, :]
        i_d += 1

        # Downwind terms along time
        G[:, :, :, :] += D_img[:,i_d,:,:,:] / grad_norms[:, :, :, :]
        G[:, :-1, :, :] += -D_img[:,i_d,1:,:,:] / grad_norms[:, 1:, :, :]
        i_d += 1

    G /= np.sqrt(2.0)

    return (tv, G)

def tv_downwind(img, mask = [], reg_z_over_reg = 1.0, reg_time = 0.0, mask_static = False, factor_reg_static = 0):
    '''
    Calculates the total variation and a subgradient of the input image img using the downwind gradient discretization

    Parameters
    ----------
    img : np.ndarray
        The array of the input image data of dimensions Nz x M x N x N
    mask : np.ndarray
        A mask used to specify regions of the image ot skip for the TV calculation.
    reg_z_over_reg : float
        The ratio of the regularization parameter in the z direction, versus the x-y plane.
    reg_time : float
        The ratio of the regularization parameter in the time direction, versus the x-y plane.

    Returns
    -------
    tv : float
        The value of the total variation.
    G : np.ndarray
        A sub-gradient array of the total variation term of same dimensions as the input image img.
    '''

    if mask != []:
        img[~mask] = 0
    Nz = img.shape[0]
    M = img.shape[1]

    D_img = pytv.tv_operators_CPU.D_downwind(img, reg_z_over_reg = reg_z_over_reg, reg_time = reg_time, mask_static = mask_static, factor_reg_static = factor_reg_static)
    tv, grad_norms = pytv.tv_operators_CPU.compute_L21_norm(D_img, return_array=True)

    # When non-differentiable, set to 0.
    grad_norms[grad_norms == 0] = np.inf

    # Careful when reading math: D_img[:,0,:,:,:] does not give r_{m,n,p,q}, but r_{m-1,n,p,q}
    G = np.zeros_like(img)
    G[:, :, :, :] += (D_img[:,0,:,:,:]+D_img[:,1,:,:,:]) / grad_norms[:, :, :, :]
    G[:, :, :-1, :] += -D_img[:,0,:,1:,:] / grad_norms[:, :, 1:, :]
    G[:, :, :, :-1] += -D_img[:,1,:,:,1:] / grad_norms[:, :, :, 1:]

    i_d = 2
    if Nz > 1 and reg_z_over_reg > 0:
        G[:, :, :, :] += D_img[:,i_d,:,:,:] / grad_norms[:, :, :, :]
        G[:-1, :, :, :] += -D_img[1:,i_d,:,:,:] / grad_norms[1:, :, :, :]
        i_d += 1

    if reg_time > 0 and M > 1:
        G[:, :, :, :] += D_img[:,i_d,:,:,:] / grad_norms[:, :, :, :]
        G[:, :-1, :, :] += -D_img[:,i_d,1:,:,:] / grad_norms[:, 1:, :, :]
        i_d += 1

    return (tv, G)

def tv_upwind(img, mask = [], reg_z_over_reg = 1.0, reg_time = 0.0, mask_static = False, factor_reg_static = 0):
    '''
    Calculates the total variation and a subgradient of the input image img using the upwind gradient discretization

    Parameters
    ----------
    img : np.ndarray
        The array of the input image data of dimensions Nz x M x N x N.
    mask : np.ndarray
        A mask used to specify regions of the image ot skip for the TV calculation.
    reg_z_over_reg : float
        The ratio of the regularization parameter in the z direction, versus the x-y plane.

    Returns
    -------
    tv : float
        The value of the total variation.
    G : np.ndarray
        A sub-gradient array of the total variation term of same dimensions as the input image img.
    '''

    if mask != []:
        img[~mask] = 0
    Nz = img.shape[0]
    M = img.shape[1]

    D_img = pytv.tv_operators_CPU.D_upwind(img, reg_z_over_reg = reg_z_over_reg, reg_time = reg_time, mask_static = mask_static, factor_reg_static = factor_reg_static)
    tv, grad_norms = pytv.tv_operators_CPU.compute_L21_norm(D_img, return_array=True)

    # When non-differentiable, set to 0.
    grad_norms[grad_norms == 0] = np.inf

    G = np.zeros_like(img)
    G[:, :, :, :] += - (D_img[:,0,:,:,:]+D_img[:,1,:,:,:]) / grad_norms[:, :, :, :]
    G[:, :, 1:, :] += D_img[:,0,:,:-1,:] / grad_norms[:, :, :-1, :]
    G[:, :, :, 1:] += D_img[:,1,:,:,:-1] / grad_norms[:, :, :, :-1]

    i_d = 2
    if Nz > 1 and reg_z_over_reg > 0:
        G[:, :, :, :] += - D_img[:,i_d,:,:,:] / grad_norms[:, :, :, :]
        G[1:, :, :, :] += D_img[:-1,i_d,:,:,:] / grad_norms[:-1, :, :, :]
        i_d += 1

    if reg_time > 0 and M > 1:
        G[:, :, :, :] += - D_img[:,i_d,:,:,:] / grad_norms[:, :, :, :]
        G[:, 1:, :, :] += D_img[:,i_d,:-1,:,:] / grad_norms[:, :-1, :, :]
        i_d += 1

    return (tv, G)

def tv_central(img, mask = [], reg_z_over_reg = 1.0, reg_time = 0.0, mask_static = False, factor_reg_static = 0):
    '''
    Calculates the total variation and a subgradient of the input image img using the central gradient discretization

    Parameters
    ----------
    img : np.ndarray
        The array of the input image data of dimensions N x N, or Nz x N x N.
    mask : np.ndarray
        A mask used to specify regions of the image ot skip for the TV calculation.
    reg_z_over_reg : float
        The ratio of the regularization parameter in the z direction, versus the x-y plane.

    Returns
    -------
    tv : float
        The value of the total variation.
    G : np.ndarray
        A sub-gradient array of the total variation term of same dimensions as the input image img.
    '''

    if mask != []:
        img[~mask] = 0
    Nz = img.shape[0]
    M = img.shape[1]

    D_img = pytv.tv_operators_CPU.D_central(img, reg_z_over_reg = reg_z_over_reg, reg_time = reg_time, mask_static = mask_static, factor_reg_static = factor_reg_static)
    tv, grad_norms = pytv.tv_operators_CPU.compute_L21_norm(D_img, return_array=True)

    # When non-differentiable, set to 0.
    grad_norms[grad_norms == 0] = np.inf

    G = np.zeros_like(img)
    G[:, :, 1:, :] += D_img[:,0,:,:-1,:] / grad_norms[:, :, :-1, :]
    G[:, :, :-1, :] += - D_img[:,0,:,1:,:] / grad_norms[:, :, 1:, :]

    G[:, :, :, 1:] += D_img[:,1,:,:,:-1] / grad_norms[:, :, :, :-1]
    G[:, :, :, :-1] += - D_img[:,1,:,:,1:] / grad_norms[:, :, :, 1:]

    i_d = 2
    if Nz > 1 and reg_z_over_reg > 0:
        if Nz == 2: # Use upwind scheme instead
            G[1:, :, :, :] += D_img[:-1,i_d,:,:,:] / grad_norms[:-1, :, :, :]
            G[:, :, :, :] += - D_img[:,i_d,:,:,:] / grad_norms[:, :, :, :]
            i_d += 1
        else:
            G[1:, :, :, :] += D_img[:-1,i_d,:,:,:] / grad_norms[:-1, :, :, :]
            G[:-1, :, :, :] += - D_img[1:,i_d,:,:,:] / grad_norms[1:, :, :, :]
        i_d += 1

    if reg_time > 0 and M > 1:
        if M == 2: # Use upwind scheme instead
            G[:, 1:, :, :] += D_img[:,i_d,:-1,:,:] / grad_norms[:, :-1, :, :]
            G[:, :, :, :] += - D_img[:,i_d,:,:,:] / grad_norms[:, :, :, :]
        else:
            G[:, 1:, :, :] += D_img[:,i_d,:-1,:,:] / grad_norms[:, :-1, :, :]
            G[:, :-1, :, :] += - D_img[:,i_d,1:,:,:] / grad_norms[:, 1:, :, :]

        i_d += 1

    G /= 2.0

    return (tv, G)
