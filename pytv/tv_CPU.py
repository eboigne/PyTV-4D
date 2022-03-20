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


import numpy as np
# import pytv.tv_2d_CPU
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

    # The intensity differences across rows.
    row_diff = np.zeros_like(img)
    if Nz > 1:
        row_diff[:-1, :, :-1, :-1] = img[:-1, :, 1:, :-1] - img[:-1, :, :-1, :-1]
    else:
        row_diff[:, :, :-1, :-1] = img[:, :, 1:, :-1] - img[:, :, :-1, :-1]

    # The intensity differences across columns.
    col_diff = np.zeros_like(img)
    if Nz > 1:
        col_diff[:-1, :, :-1, :-1] = img[:-1, :, :-1, 1:] - img[:-1, :, :-1, :-1]
    else:
        col_diff[:, :, :-1, :-1] = img[:, :, :-1, 1:] - img[:, :, :-1, :-1]

    # The intensity differences across slices.
    slice_diff = np.zeros_like(img)
    if Nz > 1:
        slice_diff[:-1, :, :-1, :-1] = np.sqrt(reg_z_over_reg) * (img[1:, :, :-1, :-1] - img[:-1, :, :-1, :-1])

    # The intensity differences across times.
    time_diff = np.zeros_like(img)
    # if Nz > 1: # TODO: No distinction in Nz > 1 is better for reg_time. Carry over this choice to DW.
        # time_diff[:-1, :-1, :-1, :-1] = np.sqrt(reg_time) * (img[:-1, 1:, :-1, :-1] - img[:-1, :-1, :-1, :-1])
    # else:
    time_diff[:, :-1, :-1, :-1] = np.sqrt(reg_time) * (img[:, 1:, :-1, :-1] - img[:, :-1, :-1, :-1])

    #  Compute the total variation.
    eps = 0
    grad_norms = np.zeros_like(img)

    if Nz > 1:
        if match_2D_form:
            grad_norms[:-1, :, :-1, :-1] = np.square(row_diff[:-1, :, :-1, :-1]) + np.square(col_diff[:-1, :, :-1, :-1]) \
                                                + np.square(slice_diff[:-1, :, :-1, :-1]) + np.square(row_diff[:-1, :, :-1, 1:]) \
                                                + np.square(col_diff[:-1, :, 1:, :-1]) + np.square(slice_diff[:-1, :, 1:, 1:]) + eps
            grad_norms[:, :-1, :-1, :-1] += np.square(time_diff[:, :-1, :-1, :-1])
            grad_norms[:, :-1, :-2, :-2] += np.square(time_diff[:, :-1, 1:-1, 1:-1])
            grad_norms = np.sqrt(grad_norms) / np.sqrt(2)
        else:
            grad_norms[:-1, :, :-1, :-1] = np.square(row_diff[:-1, :, :-1, :-1]) + np.square(col_diff[:-1, :, :-1, :-1]) \
                                                + np.square(slice_diff[:-1, :, :-1, :-1]) \
                                                + np.square(row_diff[1:, :, :-1, 1:]) + np.square(col_diff[1:, :, 1:, :-1]) \
                                                + np.square(slice_diff[:-1, :, 1:, 1:]) + eps
            grad_norms[:, :-1, :-1, :-1] += np.square(time_diff[:, :-1, :-1, :-1])
            grad_norms[:, :-1, :-2, :-2] += np.square(time_diff[:, :-1, 1:-1, 1:-1])
            # grad_norms[:, 1:, 1:-1, 1:-1] += np.square(time_diff[:, :-1, 1:-1, 1:-1])
            grad_norms = np.sqrt(grad_norms) / np.sqrt(2)
    else:
        grad_norms += np.square(row_diff)+np.square(col_diff)+np.square(slice_diff)+np.square(time_diff)
        grad_norms[:, :, :-1, :-1] += np.square(row_diff[:, :, :-1, 1:])+np.square(col_diff[:, :, 1:, :-1])+np.square(slice_diff[:, :, 1:, 1:])+eps
        grad_norms[:, :-1, :-2, :-2] += np.square(time_diff[:, :-1, 1:-1, 1:-1])
        # grad_norms[:, 1:, 1:-1, 1:-1] += np.square(time_diff[:, :-1, 1:-1, 1:-1])

        grad_norms = np.sqrt(grad_norms) / np.sqrt(2)

    tv = np.sum(grad_norms)


    # Find a subgradient.
    G = np.zeros_like(img)
    # When non-differentiable, set to 0.
    grad_norms[grad_norms == 0] = np.inf # not necessary if eps > 0

    #TODO: Check time_diff analytically. Numeric convergence test is ok
    #DW
    if Nz > 1:
        G[1:, :, 1:, 1:] += (row_diff[1:, :, :-1, 1:] + col_diff[1:, :, 1:, :-1] + slice_diff[:-1, :, 1:, 1:]) / grad_norms[:-1, :, :-1, :-1]
        G[1:, 1:, 1:, 1:] += time_diff[1:, :-1, 1:, 1:] / grad_norms[:-1, :-1, :-1, :-1]
        G[1:, :, :-1, 1:] += - row_diff[1:, :, :-1, 1:] / grad_norms[:-1, :, :-1, :-1]
        G[1:, :, 1:, :-1] += - col_diff[1:, :, 1:, :-1] / grad_norms[:-1, :, :-1, :-1]
        G[:-1, :, 1:, 1:] += - slice_diff[:-1, :, 1:, 1:] / grad_norms[:-1, :, :-1, :-1]
        G[1:, :, 1:, 1:] += - time_diff[1:, :, 1:, 1:] / grad_norms[:-1, :, :-1, :-1]
    else:
        G[:, :, 1:, 1:] += (row_diff[:, :, :-1, 1:] + col_diff[:, :, 1:, :-1]) / grad_norms[:, :, :-1, :-1]
        G[:, 1:, 1:, 1:] += time_diff[:, :-1, 1:, 1:] / grad_norms[:, :-1, :-1, :-1]
        G[:, :, :-1, 1:] += - row_diff[:, :, :-1, 1:] / grad_norms[:, :, :-1, :-1]
        G[:, :, 1:, :-1] += - col_diff[:, :, 1:, :-1] / grad_norms[:, :, :-1, :-1]
        G[:, :, 1:, 1:] += - time_diff[:, :, 1:, 1:] / grad_norms[:, :, :-1, :-1]

    # UW
    if Nz > 1:
        G[:-1, :, :-1, :-1] += - (row_diff+col_diff+slice_diff+time_diff)[:-1, :, :-1, :-1] / grad_norms[:-1, :, :-1, :-1]
        G[:-1, :, 1:, :-1] += row_diff[:-1, :, :-1, :-1] / grad_norms[:-1, :, :-1, :-1]
        G[:-1, :, :-1, 1:] += col_diff[:-1, :, :-1, :-1] / grad_norms[:-1, :, :-1, :-1]
        G[1:, :, :-1, :-1] += slice_diff[:-1, :, :-1, :-1] / grad_norms[:-1, :, :-1, :-1]
        G[:-1, 1:, :-1, :-1] += time_diff[:-1, 1:, :-1, :-1] / grad_norms[:-1, 1:, :-1, :-1]
    else:
        G[:, :, :-1, :-1] += - (row_diff+col_diff+slice_diff+time_diff)[:, :, :-1, :-1] / grad_norms[:, :, :-1, :-1]
        G[:, :, 1:, :-1] += row_diff[:, :, :-1, :-1] / grad_norms[:, :, :-1, :-1]
        G[:, :, :-1, 1:] += col_diff[:, :, :-1, :-1] / grad_norms[:, :, :-1, :-1]
        # G[:, :, :-1, :-1] += slice_diff[:, :, :-1, :-1] / grad_norms[:, :, :-1, :-1] # useless, slice_diff is 0 for Nz = 1
        G[:, 1:, :-1, :-1] += time_diff[:, 1:, :-1, :-1] / grad_norms[:, 1:, :-1, :-1]

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
    grad_norms[grad_norms == 0] = np.inf # not necessary if eps > 0

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
    grad_norms[grad_norms == 0] = np.inf # not necessary if eps > 0

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

def tv_centered(img, mask = [], reg_z_over_reg = 1.0):
    '''
    Calculates the total variation and a subgradient of the input image img using the centered gradient discretization

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

    if len(img.shape) == 2:
        return(pytv.tv_2d_CPU.tv_centered(img))
    elif (len(img.shape) == 3 and img.shape[0] < 3):
        return(pytv.tv_2d_CPU.tv_centered(img[0]))

    # The intensity differences across rows.
    row_diff = 0.5 * ( img[1:-1, 2:, 1:-1] - img[1:-1, :-2, 1:-1] )

    # The intensity differences across columns.
    col_diff = 0.5 * ( img[1:-1, 1:-1, 2:] - img[1:-1, 1:-1, :-2] )

    # The intensity differences across slices.
    slice_diff = np.sqrt(reg_z_over_reg) * 0.5 * (img[2:, 1:-1, 1:-1] - img[:-2, 1:-1, 1:-1])

    #  Compute the total variation.
    eps = 0
    grad_norms = np.sqrt(np.square(row_diff)+np.square(col_diff)+np.square(slice_diff)+eps)
    tv = np.sum(grad_norms)

    # Find a subgradient.
    G = np.zeros_like(img)
    # When non-differentiable, set to 0.
    grad_norms[grad_norms == 0] = np.inf # not necessary if eps > 0

    G[1:-1, 2:, 1:-1] += row_diff/grad_norms
    G[1:-1, :-2, 1:-1] += - row_diff/grad_norms
    G[1:-1, 1:-1, 2:] += col_diff/grad_norms
    G[1:-1, 1:-1, :-2] += - col_diff/grad_norms
    G[2:, 1:-1, 1:-1] += slice_diff/grad_norms
    G[:-2, 1:-1, 1:-1] += - slice_diff/grad_norms

    return (tv, G)
