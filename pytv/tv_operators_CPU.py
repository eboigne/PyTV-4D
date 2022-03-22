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

def compute_L21_norm(D_img, return_array = False):
    '''
    Compute the L2,1 norm of an image of discrete differences: |x|_2,1 = \sum_i \sqrt(\sum_j x_{i,j}^2),
    with index i summing over image pixels, and index j summing over the difference terms.
    Usage: TV(img) = reg * compute_L21_norm(D(img))

    Parameters
    ----------
    D_img : np.ndarray
        The numpy array of the discrete gradient of dimensions Nz x Nd x M x N x N.

    Returns
    -------
    float
        The L2,1 norm of the given input array.
    '''

    out = np.square(D_img)
    out = np.sum(out, axis = 1)
    out = np.sqrt(out)
    out_sum = np.sum(out)

    if return_array:
        return(out_sum, out)
    else:
        return(out_sum)

def D_hybrid(img, reg_z_over_reg = 1.0, reg_time = 0, mask_static = False, factor_reg_static = 0):
    '''
    Calculates the output of the input image img by the operator D (gradient discretized using hybrid scheme)

    Parameters
    ----------
    img : np.ndarray
        The array of the input image data of dimensions Nz x M x N x N.
    reg_z_over_reg : float
        The ratio of the regularization parameter in the z direction, versus the x-y plane.
    reg_time : float
        The ratio of the regularization parameter in the time direction, versus the x-y plane.
    mask_static : np.ndarray
        An of dimensions 1 x 1 x N x N serving as a mask to indicate pixels on which to enforce a different
        time regularization parameter, for instance used to enforce more static regions in the image.
    factor_reg_static : float
        The regularization parameter to compute in the region of the image specified by mask_static.

    Returns
    -------
    np.ndarray
        The array of the discretized gradient D(img) of dimensions Nz x Nd x M x N x N.
    '''

    if reg_z_over_reg == np.nan:
        reg_z_over_reg = 0.0

    Nz = img.shape[0]
    M = img.shape[1]
    N = img.shape[-1]

    N_d = 4
    if Nz > 1 and reg_z_over_reg > 0:
        N_d += 2
    if reg_time > 0 and M > 1:
        N_d += 2
    D_img = np.zeros([Nz, N_d, M, N, N])

    # The intensity differences across rows (Upwind / Forward)
    D_img[:, 0, :, :-1, :] = img[:, :, 1:, :] - img[:, :, :-1, :]

    # The intensity differences across columns (Upwind / Forward)
    D_img[:, 1, :, :, :-1] = img[:, :, :, 1:] - img[:, :, :, :-1]

    # The intensity differences across rows (Downwind / Backward)
    D_img[:, 2, :, 1:, :] = D_img[:, 0, :, :-1, :]

    # The intensity differences across columns (Downwind / Backward)
    D_img[:, 3, :, :, 1:] = D_img[:, 1, :, :, :-1]

    i_d = 4
    if Nz > 1 and reg_z_over_reg > 0:

        # The intensity differences across slices (Upwind / Forward)
        D_img[:-1, i_d, :, :, :] = np.sqrt(reg_z_over_reg) * (img[1:, :, :, :] - img[:-1, :, :, :])
        i_d += 1

        # The intensity differences across z (Downwind / Backward)
        D_img[1:, i_d, :, :, :] = D_img[:-1, i_d-1, :, :, :]
        i_d += 1

    if reg_time > 0 and M > 1:

        # The intensity differences across times (Upwind / Forward)
        D_img[:, i_d, :-1, :, :] = np.sqrt(reg_time) * (img[:, 1:, :, :] - img[:, :-1, :, :])

        # The intensity differences across time (Downwind / Backward)
        D_img[:, i_d+1, 1:, :, :] = D_img[:, i_d, :-1, :, :]

        if isinstance(mask_static, np.ndarray):
            mask_static = np.tile(mask_static, [Nz, M, 1, 1])

            D_img_temp = D_img[:,i_d,:,:,:].copy()
            D_img_temp[mask_static] *= np.sqrt(factor_reg_static)
            D_img[:,i_d,:,:,:] = D_img_temp

            D_img_temp = D_img[:,i_d+1,:,:,:].copy()
            D_img_temp[mask_static] *= np.sqrt(factor_reg_static)
            D_img[:,i_d+1,:,:,:] = D_img_temp

        i_d += 2

    return (D_img/np.sqrt(2.0))

def D_downwind(img, reg_z_over_reg = 1.0, reg_time = 0, mask_static = False, factor_reg_static = 0):
    '''
    Calculates the output of the input image img by the operator D (gradient discretized using downwind scheme)

    Parameters
    ----------
    img : np.ndarray
        The array of the input image data of dimensions Nz x M x N x N.
    reg_z_over_reg : float
        The ratio of the regularization parameter in the z direction, versus the x-y plane.
    reg_time : float
        The ratio of the regularization parameter in the time direction, versus the x-y plane.

    Returns
    -------
    np.ndarray
        The array of the discretized gradient D(img) of dimensions Nz x Nd x M x N x N.
    '''

    if reg_z_over_reg == np.nan:
        reg_z_over_reg = 0.0

    Nz = img.shape[0]
    M = img.shape[1]
    N = img.shape[-1]

    N_d = 2
    if Nz > 1 and reg_z_over_reg > 0:
        N_d += 1
    if reg_time > 0 and M > 1:
        N_d += 1

    D_img = np.zeros([Nz, N_d, M, N, N])

    # The intensity differences across rows (Downwind / Backward)
    D_img[:, 0, :, 1:, :] = (img[:, :, 1:, :] - img[:, :, :-1, :])

    # The intensity differences across columns (Downwind / Backward)
    D_img[:, 1, :, :, 1:] = (img[:, :, :, 1:] - img[:, :, :, :-1])

    i_d = 2
    if Nz > 1 and reg_z_over_reg > 0:
        # The intensity differences across z (Downwind / Backward)
        D_img[1:, i_d, :, :, :] = np.sqrt(reg_z_over_reg) * (img[1:, :, :, :] - img[:-1, :, :, :])
        i_d += 1

    if reg_time > 0 and M > 1:
        # The intensity differences across time (Downwind / Backward)
        D_img[:, i_d, 1:, :, :] = np.sqrt(reg_time) * (img[:, 1:, :, :] - img[:, :-1, :, :])

        if isinstance(mask_static, np.ndarray):
            mask_static = np.tile(mask_static, [Nz, M, 1, 1])
            D_img_temp = D_img[:,i_d,:,:,:].copy()
            D_img_temp[mask_static] *= np.sqrt(factor_reg_static)
            D_img[:,i_d,:,:,:] = D_img_temp
        i_d += 1

    return D_img

def D_upwind(img, reg_z_over_reg = 1.0, reg_time = 0, mask_static = False, factor_reg_static = 0):
    '''
    Calculates the output of the input image img by the operator D (gradient discretized using upwind scheme)

    Parameters
    ----------
    img : np.ndarray
        The array of the input image data of dimensions Nz x M x N x N.
    reg_z_over_reg : float
        The ratio of the regularization parameter in the z direction, versus the x-y plane.
    reg_time : float
        The ratio of the regularization parameter in the time direction, versus the x-y plane.

    Returns
    -------
    np.ndarray
        The array of the discretized gradient D(img) of dimensions Nz x Nd x M x N x N.
    '''

    if reg_z_over_reg == np.nan:
        reg_z_over_reg = 0.0

    Nz = img.shape[0]
    M = img.shape[1]
    N = img.shape[-1]

    N_d = 2
    if Nz > 1 and reg_z_over_reg > 0:
        N_d += 1
    if reg_time > 0 and M > 1:
        N_d += 1

    D_img = np.zeros([Nz, N_d, M, N, N])

    # The intensity differences across rows (Upwind / Forward)
    D_img[:, 0, :, :-1, :] = img[:, :, 1:, :] - img[:, :, :-1, :]

    # The intensity differences across columns (Upwind / Forward)
    D_img[:, 1, :, :, :-1] = img[:, :, :, 1:] - img[:, :, :, :-1]

    i_d = 2
    if Nz > 1 and reg_z_over_reg > 0:
        # The intensity differences across slices (Upwind / Forward)
        D_img[:-1, i_d, :, :, :] = np.sqrt(reg_z_over_reg) * (img[1:, :, :, :] - img[:-1, :, :, :])
        i_d += 1

    if reg_time > 0 and M > 1:
        # The intensity differences across times (Upwind / Forward)
        D_img[:, i_d, :-1, :, :] = np.sqrt(reg_time) * (img[:, 1:, :, :] - img[:, :-1, :, :])

        if isinstance(mask_static, np.ndarray):
            mask_static = np.tile(mask_static, [Nz, M, 1, 1])
            D_img_temp = D_img[:,i_d,:,:,:].copy()
            D_img_temp[mask_static] *= np.sqrt(factor_reg_static)
            D_img[:,i_d,:,:,:] = D_img_temp
        i_d += 1

    return D_img

def D_central(img, reg_z_over_reg = 1.0, reg_time = 0, mask_static = False, factor_reg_static = 0):
    '''
    Calculates the output of the input image img by the operator D (gradient discretized using central scheme)

    Parameters
    ----------
    img : np.ndarray
        The array of the input image data of dimensions Nz x M x N x N.
    reg_z_over_reg : float
        The ratio of the regularization parameter in the z direction, versus the x-y plane.
    reg_time : float
        The ratio of the regularization parameter in the time direction, versus the x-y plane.

    Returns
    -------
    np.ndarray
        The array of the discretized gradient D(img) of dimensions Nz x Nd x M x N x N.
    '''

    if reg_z_over_reg == np.nan:
        reg_z_over_reg = 0.0

    Nz = img.shape[0]
    M = img.shape[1]
    N = img.shape[-1]

    N_d = 2
    if Nz > 2 and reg_z_over_reg > 0:
        N_d += 1
    if reg_time > 0 and M > 1:
        N_d += 1

    D_img = np.zeros([Nz, N_d, M, N, N])

    # The intensity differences across rows
    D_img[:,0,:,1:-1,:] = (img[:,:, 2:,:] - img[:,:, :-2,:])

    # The intensity differences across columns
    D_img[:,1,:,:,1:-1] = (img[:,:,:,2:] - img[:,:,:,:-2])
    
    i_d = 2
    # The intensity differences across slices
    if Nz > 1 and reg_z_over_reg > 0:
        if Nz == 2: # Use upwind scheme instead
            D_img[:-1, i_d, :, :, :] = np.sqrt(reg_z_over_reg) * (img[1:, :, :, :] - img[:-1, :, :, :])
        else:
            D_img[1:-1,i_d,:,:,:] = np.sqrt(reg_z_over_reg) * (img[2:,:, :, :] - img[:-2,:, :, :])
        i_d += 1

    # The intensity differences across times
    if reg_time > 0 and M > 1:
        if M == 2: # Use upwind scheme instead
            D_img[:, i_d, :-1, :, :] = np.sqrt(reg_time) * (img[:, 1:, :, :] - img[:, :-1, :, :])
        else:
            D_img[:,i_d,1:-1,:,:] = np.sqrt(reg_time) * (img[:,2:,:,:] - img[:,:-2,:,:])

        if isinstance(mask_static, np.ndarray):
            mask_static = np.tile(mask_static, [Nz, M, 1, 1])
            D_img_temp = D_img[:,i_d,:,:,:].copy()
            D_img_temp[mask_static] *= np.sqrt(factor_reg_static)
            D_img[:,i_d,:,:,:] = D_img_temp
        i_d += 1

    return (D_img / 2.0)

def D_T_hybrid(img, reg_z_over_reg = 1.0, reg_time = 0, mask_static = False, factor_reg_static = 0):
    '''
    Calculates the output of the input image img by the operator D^T (tranposed gradient discretized using hybrid scheme)

    Parameters
    ----------
    img : np.ndarray
        The array of the input image data of dimensions Nz x Nd x M x N x N.
    reg_z_over_reg : float
        The ratio of the regularization parameter in the z direction, versus the x-y plane.
    reg_time : float
        The ratio of the regularization parameter in the time direction, versus the x-y plane.
    mask_static : np.ndarray
        An of dimensions 1 x 1 x N x N serving as a mask to indicate pixels on which to enforce a different
        time regularization parameter, for instance used to enforce more static regions in the image.
    factor_reg_static : float
        The regularization parameter to compute in the region of the image specified by mask_static.

    Returns
    -------
    np.ndarray
        The array of the discretized gradient D^T(img) of dimensions Nz x M x N x N.
    '''

    if reg_z_over_reg == np.nan:
        reg_z_over_reg = 0.0

    Nz = img.shape[0]
    Nd = img.shape[1]
    M = img.shape[2]
    N = img.shape[-1]

    D_T_img = np.zeros([Nz, M, N, N])


    # Forward row term
    D_T_img[:,:,1:,:] += img[:,0,:,:-1,:]
    D_T_img[:,:,:-1,:] += -img[:,0,:,:-1,:]

    # # Forward col term
    D_T_img[:,:,:,1:] += img[:,1,:,:,:-1]
    D_T_img[:,:,:,:-1] += -img[:,1,:,:,:-1]

    # Backward row term
    D_T_img[:,:,1:,:] += img[:, 2, :, 1:, :]
    D_T_img[:,:,:-1,:] += -img[:, 2, :, 1:, :]

    # Backward col term
    D_T_img[:,:,:,1:] += img[:, 3, :, :, 1:]
    D_T_img[:,:,:,:-1] += -img[:, 3, :, :, 1:]

    i_d = 4
    # The intensity differences across slices
    if Nz > 1 and reg_z_over_reg > 0:

        # Forward z terms
        D_T_img[1:,:,:,:] += np.sqrt(reg_z_over_reg) * img[:-1, i_d, :, :, :]
        D_T_img[:-1,:,:,:] += - np.sqrt(reg_z_over_reg) * img[:-1, i_d, :, :, :]
        i_d += 1

        # Backward z terms
        D_T_img[1:,:,:,:] += np.sqrt(reg_z_over_reg) * img[1:, i_d,:,:,:]
        D_T_img[:-1,:,:,:] += -np.sqrt(reg_z_over_reg) * img[1:, i_d,:,:,:]
        i_d += 1

    # The intensity differences across time
    if reg_time > 0 and M > 1:
        D_T_img_time_update = np.zeros_like(D_T_img)

        # Forward time terms
        D_T_img_time_update[:,1:,:,:] += np.sqrt(reg_time) * img[:, i_d, :-1, :, :]
        D_T_img_time_update[:,:-1,:,:] += -np.sqrt(reg_time) * img[:, i_d, :-1, :, :]
        i_d += 1

        # Backward time terms
        D_T_img_time_update[:, 1:, :, :] += np.sqrt(reg_time) * img[:, i_d, 1:, :,:]
        D_T_img_time_update[:, :-1, :, :] += - np.sqrt(reg_time) * img[:, i_d, 1:,:,:]
        i_d += 1

        if isinstance(mask_static, np.ndarray):
            mask_static = np.tile(mask_static, [Nz, M, 1, 1])
            D_T_img_time_update[mask_static] *= np.sqrt(factor_reg_static)

        D_T_img += D_T_img_time_update

    return(D_T_img/np.sqrt(2.0))

def D_T_downwind(img, reg_z_over_reg = 1.0, reg_time = 0, mask_static = False, factor_reg_static = 0):
    '''
    Calculates the output of the input image img by the operator D^T (tranposed gradient discretized using downwind scheme)

    Parameters
    ----------
    img : np.ndarray
        The array of the input image data of dimensions Nz x Nd x M x N x N.
    reg_z_over_reg : float
        The ratio of the regularization parameter in the z direction, versus the x-y plane.
    reg_time : float
        The ratio of the regularization parameter in the time direction, versus the x-y plane.

    Returns
    -------
    np.ndarray
        The array of the discretized gradient D^T(img) of dimensions Nz x M x N x N.
    '''

    if reg_z_over_reg == np.nan:
        reg_z_over_reg = 0.0

    Nz = img.shape[0]
    Nd = img.shape[1]
    M = img.shape[2]
    N = img.shape[-1]

    D_T_img = np.zeros([Nz, M, N, N])

    # Backward row term
    D_T_img[:,:,1:,:] += img[:, 0, :, 1:, :]
    D_T_img[:,:,:-1,:] += -img[:, 0, :, 1:, :]

    # Backward col term
    D_T_img[:,:,:,1:] += img[:, 1, :, :, 1:]
    D_T_img[:,:,:,:-1] += -img[:, 1, :, :, 1:]

    i_d = 2
    # The intensity differences across slices
    if Nz > 1 and reg_z_over_reg > 0:
        D_T_img[1:,:,:,:] += np.sqrt(reg_z_over_reg) * img[1:, i_d,:,:,:]
        D_T_img[:-1,:,:,:] += -np.sqrt(reg_z_over_reg) * img[1:, i_d,:,:,:]
        i_d += 1

    # The intensity differences across time
    if reg_time > 0 and M > 1:
        D_T_img_time_update = np.zeros_like(D_T_img)

        D_T_img_time_update[:, 1:, :, :] += np.sqrt(reg_time) * img[:, i_d, 1:, :,:]
        D_T_img_time_update[:, :-1, :, :] += - np.sqrt(reg_time) * img[:, i_d, 1:,:,:]
        i_d += 1

        if isinstance(mask_static, np.ndarray):
            mask_static = np.tile(mask_static, [Nz, M, 1, 1])
            D_T_img_time_update[mask_static] *= np.sqrt(factor_reg_static)

        D_T_img += D_T_img_time_update

    return(D_T_img)

def D_T_upwind(img, reg_z_over_reg = 1.0, reg_time = 0, mask_static = False, factor_reg_static = 0):
    '''
    Calculates the output of the input image img by the operator D^T (tranposed gradient discretized using upwind scheme)

    Parameters
    ----------
    img : np.ndarray
        The array of the input image data of dimensions Nz x Nd x M x N x N.
    reg_z_over_reg : float
        The ratio of the regularization parameter in the z direction, versus the x-y plane.
    reg_time : float
        The ratio of the regularization parameter in the time direction, versus the x-y plane.

    Returns
    -------
    np.ndarray
        The array of the discretized gradient D^T(img) of dimensions Nz x M x N x N.
    '''

    if reg_z_over_reg == np.nan:
        reg_z_over_reg = 0.0

    Nz = img.shape[0]
    Nd = img.shape[1]
    M = img.shape[2]
    N = img.shape[-1]
    D_T_img = np.zeros([Nz, M, N, N])

    # Forward row term
    D_T_img[:,:,1:,:] += img[:,0,:,:-1,:]
    D_T_img[:,:,:-1,:] += -img[:,0,:,:-1,:]

    # # Forward col term
    D_T_img[:,:,:,1:] += img[:,1,:,:,:-1]
    D_T_img[:,:,:,:-1] += -img[:,1,:,:,:-1]

    i_d = 2
    # The intensity differences across slices
    if Nz > 1 and reg_z_over_reg > 0:
        D_T_img[1:,:,:,:] += np.sqrt(reg_z_over_reg) * img[:-1, i_d, :, :, :]
        D_T_img[:-1,:,:,:] += - np.sqrt(reg_z_over_reg) * img[:-1, i_d, :, :, :]
        i_d += 1

    # The intensity differences across time
    if reg_time > 0 and M > 1:
        D_T_img_time_update = np.zeros_like(D_T_img)

        D_T_img_time_update[:,1:,:,:] += np.sqrt(reg_time) * img[:, i_d, :-1, :, :]
        D_T_img_time_update[:,:-1,:,:] += -np.sqrt(reg_time) * img[:, i_d, :-1, :, :]
        i_d += 1

        if isinstance(mask_static, np.ndarray):
            mask_static = np.tile(mask_static, [Nz, M, 1, 1])
            D_T_img_time_update[mask_static] *= np.sqrt(factor_reg_static)

        D_T_img += D_T_img_time_update

    return(D_T_img)

def D_T_central(img, reg_z_over_reg = 1.0, reg_time = 0, mask_static = False, factor_reg_static = 0):
    '''
    Calculates the output of the input image img by the operator D^T (tranposed gradient discretized using central scheme)

    Parameters
    ----------
    img : np.ndarray
        The array of the input image data of dimensions Nz x Nd x M x N x N.
    reg_z_over_reg : float
        The ratio of the regularization parameter in the z direction, versus the x-y plane.
    reg_time : float
        The ratio of the regularization parameter in the time direction, versus the x-y plane.

    Returns
    -------
    np.ndarray
        The array of the discretized gradient D^T(img) of dimensions Nz x M x N x N.
    '''

    if reg_z_over_reg == np.nan:
        reg_z_over_reg = 0.0

    Nz = img.shape[0]
    N_d = img.shape[1]
    N = img.shape[-1]
        
    M = img.shape[2]
    D_T_img = np.zeros([Nz, M, N, N])

    # The intensity differences across rows
    D_T_img[:,:, 2:,:] += img[:,0,:,1:-1,:]
    D_T_img[:,:, :-2,:] += -img[:,0,:,1:-1,:]

    # The intensity differences across columns
    D_T_img[:,:, :,2:] += img[:,1,:,:,1:-1]
    D_T_img[:,:, :,:-2] += -img[:,1,:,:,1:-1]

    i_d = 2
    # The intensity differences across slices
    if Nz > 1 and reg_z_over_reg > 0:
        if Nz == 2: # Use upwind scheme instead
            D_T_img[1:,:,:,:] += np.sqrt(reg_z_over_reg) * img[:-1, i_d, :, :, :]
            D_T_img[:-1,:,:,:] += -np.sqrt(reg_z_over_reg) * img[:-1, i_d, :, :, :]
        else:
            D_T_img[2:,:,:,:] += np.sqrt(reg_z_over_reg) * img[1:-1, i_d, :, :, :]
            D_T_img[:-2,:,:,:] += -np.sqrt(reg_z_over_reg) * img[1:-1, i_d, :, :, :]
        i_d += 1

    # The intensity differences across times
    if reg_time > 0 and M > 1:
        D_T_img_time_update = np.zeros_like(D_T_img)

        if M == 2: # Use upwind scheme instead
            D_T_img_time_update[:, 1:, :, :] += np.sqrt(reg_time) * img[:, i_d, :-1, :, :]
            D_T_img_time_update[:, :-1, :, :] += - np.sqrt(reg_time) * img[:, i_d, :-1, :, :]
        else:
            D_T_img_time_update[:, 2:, :, :] += np.sqrt(reg_time) * img[:, i_d, 1:-1, :, :]
            D_T_img_time_update[:, :-2, :, :] += - np.sqrt(reg_time) * img[:, i_d, 1:-1, :, :]

        if isinstance(mask_static, np.ndarray):
            mask_static = np.tile(mask_static, [Nz, M, 1, 1])
            D_T_img_time_update[mask_static] *= np.sqrt(factor_reg_static)
        D_T_img += D_T_img_time_update
        i_d += 1

    return(D_T_img / 2.0)
