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
import torch

def compute_L21_norm(D_img, return_array = False):
    '''
    Compute the L2,1 norm of an image of discrete differences: |x|_2,1 = \sum_i \sqrt(\sum_j x_{i,j}^2),
    with index i summing over image pixels, and index j summing over the difference terms.
    Usage: TV(img) = reg * compute_L21_norm(D(img))

    Parameters
    ----------
    D_img : np.ndarray or torch.Tensor
        The array of the discrete gradient of dimensions Nz x Nd x M x N x N.
    return_array : boolean
        Whether to return the array of the L2 norms.

    Returns
    -------
    l21_norm : float
        The value of the L2,1 norm.
    out : np.ndarray or torch.Tensor
        An array of the L2 norms of size Nz x M x N x N.
    '''

    if ~isinstance(D_img, torch.Tensor):
        D_img = torch.as_tensor(D_img).cuda()
    else:
        D_img = D_img.cuda()

    try:
        out = torch.square(D_img)
    except:
        out = D_img * D_img

    out = torch.sum(out, axis = 1)
    out = torch.sqrt(out)
    l21_norm = torch.sum(out)

    if return_array:
        return(l21_norm.cpu().detach().numpy(), out)
    else:
        del out
        return(l21_norm.cpu().detach().numpy())

def D_hybrid(img, reg_z_over_reg = 1.0, reg_time = 0, mask_static = False, factor_reg_static = 0, return_pytorch_tensor = False):
    '''
    Calculates the output of the input image img by the operator D (gradient discretized using hybrid scheme)

    Parameters
    ----------
    img : np.ndarray
        The array of the input image data of dimensions Nz x M x N x N.
    reg_z_over_reg : float
        The ratio of the regularization parameter in the z direction, versus the x-y plane.
    reg_time : float
        The ratio (\mu) of the regularization parameter in the time direction, versus the x-y plane.
    mask_static : np.ndarray
        An array of dimensions 1 x 1 x N x N serving as a mask to indicate pixels on which to enforce a different
        time regularization parameter, for instance used to enforce more static regions in the image.
    factor_reg_static : float
        The regularization parameter to compute in the region of the image specified by mask_static.
    return_pytorch_tensor : boolean
        Whether to return a numpy np.ndarray or a PyTorch torch.Tensor

    Returns
    -------
    np.ndarray or torch.Tensor
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
    D_img = torch.zeros([Nz, N_d, M, N, N]).cuda()

    # Reshape input as (M, 1, Nz, N, N), such that can use conv3d directly on Nz (typically Nz >> M).
    img_tensor = np.reshape(img, (1,)+img.shape).astype('float32') # (1, Nz, M, N, N)
    img_tensor = torch.as_tensor(np.transpose(img_tensor.astype('float32'), [2, 0, 1, 3, 4])).cuda() # (M, 1, Nz, N, N)

    kernel_col = np.array([[[-1,1]]]).astype('float32')
    kernel_col = torch.as_tensor(np.reshape(kernel_col, (1,1)+kernel_col.shape)).cuda()

    kernel_row = np.array([[[-1],[1]]]).astype('float32')
    kernel_row = torch.as_tensor(np.reshape(kernel_row, (1,1)+kernel_row.shape)).cuda()

    kernel_slice = np.array([[[-1]],[[1]]]).astype('float32')
    kernel_slice = torch.as_tensor(np.reshape(kernel_slice, (1,1)+kernel_slice.shape)).cuda()

    # The intensity differences across rows (Upwind / Forward)
    # (Transpose to switch M and Nz)
    D_img[:, 0, :, :-1, :] = torch.transpose(torch.nn.functional.conv3d(img_tensor, kernel_row, bias=None, stride=1, padding = 0)[:, 0, :, :, :], 1, 0)

    # The intensity differences across columns (Upwind / Forward)
    # (Transpose to switch M and Nz)
    D_img[:, 1, :, :, :-1] = torch.transpose(torch.nn.functional.conv3d(img_tensor, kernel_col, bias=None, stride=1, padding = 0)[:, 0, :, :, :], 1, 0)

    # The intensity differences across rows (Downwind / Backward)
    # (Transpose to switch M and Nz)
    D_img[:, 2, :, 1:, :] = D_img[:, 0, :, :-1, :]

    # The intensity differences across columns (Downwind / Backward)
    # (Transpose to switch M and Nz)
    D_img[:, 3, :, :, 1:] = D_img[:, 1, :, :, :-1]

    i_d = 4
    if Nz > 1 and reg_z_over_reg > 0:
        # The intensity differences across slices (Upwind / Forward) (Transpose to switch M and Nz)
        D_img[:-1, i_d, :, :, :] = np.sqrt(reg_z_over_reg) * torch.transpose(torch.nn.functional.conv3d(img_tensor, kernel_slice, bias=None, stride=1, padding = 0)[:, 0, :, :, :], 1, 0)
        i_d += 1

        # The intensity differences across slices (Downwind / Backward) (Transpose to switch M and Nz)
        D_img[1:, i_d, :, :, :] = D_img[:-1, i_d-1, :, :, :]
        i_d += 1

    # Reshape from (M, 1, Nz, N, N) to (Nz, M, N, N)
    img_tensor = torch.transpose(img_tensor, 0, 2)[:,0,:,:,:]

    # Given that M is usually <10, it's not worth using the convolution operator there
    if reg_time > 0 and M > 1:

        # The intensity differences across times (Upwind / Forward)
        D_img[:, i_d, :-1, :, :] = np.sqrt(reg_time) * (img_tensor[:, 1:, :, :] - img_tensor[:, :-1, :, :])

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

    del img_tensor, kernel_row, kernel_col, kernel_slice

    if not return_pytorch_tensor:
        D_img2 = D_img.cpu().detach().numpy()
        del D_img
        D_img = D_img2

    return (D_img/np.sqrt(2.0))

def D_downwind(img, reg_z_over_reg = 1.0, reg_time = 0, mask_static = False, factor_reg_static = 0, return_pytorch_tensor = False):
    '''
    Calculates the output of the input image img by the operator D (gradient discretized using downwind scheme)

    Parameters
    ----------
    img : np.ndarray
        The array of the input image data of dimensions Nz x M x N x N.
    reg_z_over_reg : float
        The ratio of the regularization parameter in the z direction, versus the x-y plane.
    reg_time : float
        The ratio (\mu) of the regularization parameter in the time direction, versus the x-y plane.
    mask_static : np.ndarray
        An array of dimensions 1 x 1 x N x N serving as a mask to indicate pixels on which to enforce a different
        time regularization parameter, for instance used to enforce more static regions in the image.
    factor_reg_static : float
        The regularization parameter to compute in the region of the image specified by mask_static.
    return_pytorch_tensor : boolean
        Whether to return a numpy np.ndarray or a PyTorch torch.Tensor

    Returns
    -------
    np.ndarray or torch.Tensor
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
    i_d = 2

    D_img = torch.zeros([Nz, N_d, M, N, N]).cuda()

    # Reshape input as (M, 1, Nz, N, N), such that can use conv3d directly on Nz (typically Nz >> M).
    img_tensor = np.reshape(img, (1,)+img.shape).astype('float32') # (1, Nz, M, N, N)
    img_tensor = torch.as_tensor(np.transpose(img_tensor.astype('float32'), [2, 0, 1, 3, 4])).cuda() # (M, 1, Nz, N, N)

    kernel_col = np.array([[[-1,1]]]).astype('float32')
    kernel_col = torch.as_tensor(np.reshape(kernel_col, (1,1)+kernel_col.shape)).cuda()

    kernel_row = np.array([[[-1],[1]]]).astype('float32')
    kernel_row = torch.as_tensor(np.reshape(kernel_row, (1,1)+kernel_row.shape)).cuda()

    kernel_slice = np.array([[[-1]],[[1]]]).astype('float32')
    kernel_slice = torch.as_tensor(np.reshape(kernel_slice, (1,1)+kernel_slice.shape)).cuda()

    # The intensity differences across rows (Downwind / Backward)
    # (Transpose to switch M and Nz)
    D_img[:, 0, :, 1:, :] = torch.transpose(torch.nn.functional.conv3d(img_tensor, kernel_row, bias=None, stride=1, padding = 0)[:, 0, :, :, :], 1, 0)

    # The intensity differences across columns (Downwind / Backward)
    # (Transpose to switch M and Nz)
    D_img[:, 1, :, :, 1:] = torch.transpose(torch.nn.functional.conv3d(img_tensor, kernel_col, bias=None, stride=1, padding = 0)[:, 0, :, :, :], 1, 0)

    # The intensity differences across slices (Downwind / Backward)
    # (Transpose to switch M and Nz)
    i_d = 2
    if Nz > 1 and reg_z_over_reg > 0:
        D_img[1:, i_d, :, :, :] = np.sqrt(reg_z_over_reg) * torch.transpose(torch.nn.functional.conv3d(img_tensor, kernel_slice, bias=None, stride=1, padding = 0)[:, 0, :, :, :], 1, 0)
        i_d += 1

    # Reshape from (M, 1, Nz, N, N) to (Nz, M, N, N)
    img_tensor = torch.transpose(img_tensor, 0, 2)[:,0,:,:,:]

    # The intensity differences across times (Downwind / Backward)
    # Given that M is usually <10, it's not worth using the convolution operator there
    if reg_time > 0 and M > 1:

        # D_img[:, i_d, 1:, :, :] = np.sqrt(reg_time) * torch.reshape(torch.as_tensor((img[:, 1:, :, :] - img[:, :-1, :, :])), [Nz, M-1, N, N])
        D_img[:, i_d, 1:, :, :] = np.sqrt(reg_time) * (img_tensor[:, 1:, :, :] - img_tensor[:, :-1, :, :])
        # D_img[:, i_d, 1:, :, :] = np.sqrt(reg_time) * torch.nn.functional.conv3d(img_tensor, kernel_slice, bias=None, stride=1, padding = 0)[:, 0, :, :, :]

        if isinstance(mask_static, np.ndarray):
            mask_static = np.tile(mask_static, [Nz, M, 1, 1])
            D_img_temp = D_img[:,i_d,:,:,:].copy()
            D_img_temp[mask_static] *= np.sqrt(factor_reg_static)
            D_img[:,i_d,:,:,:] = D_img_temp
        i_d += 1

    del img_tensor, kernel_row, kernel_col, kernel_slice

    if not return_pytorch_tensor:
        D_img2 = D_img.cpu().detach().numpy()
        del D_img
        D_img = D_img2

    return D_img

def D_upwind(img, reg_z_over_reg = 1.0, reg_time = 0, mask_static = False, factor_reg_static = 0, return_pytorch_tensor = False):
    '''
    Calculates the output of the input image img by the operator D (gradient discretized using upwind scheme)

    Parameters
    ----------
    img : np.ndarray
        The array of the input image data of dimensions Nz x M x N x N.
    reg_z_over_reg : float
        The ratio of the regularization parameter in the z direction, versus the x-y plane.
    reg_time : float
        The ratio (\mu) of the regularization parameter in the time direction, versus the x-y plane.
    mask_static : np.ndarray
        An array of dimensions 1 x 1 x N x N serving as a mask to indicate pixels on which to enforce a different
        time regularization parameter, for instance used to enforce more static regions in the image.
    factor_reg_static : float
        The regularization parameter to compute in the region of the image specified by mask_static.
    return_pytorch_tensor : boolean
        Whether to return a numpy np.ndarray or a PyTorch torch.Tensor

    Returns
    -------
    np.ndarray or torch.Tensor
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
    i_d = 2

    D_img = torch.zeros([Nz, N_d, M, N, N]).cuda()

    # Reshape input as (M, 1, Nz, N, N), such that can use conv3d directly on Nz (typically Nz >> M).
    img_tensor = np.reshape(img, (1,)+img.shape).astype('float32') # (1, Nz, M, N, N)
    img_tensor = torch.as_tensor(np.transpose(img_tensor.astype('float32'), [2, 0, 1, 3, 4])).cuda() # (M, 1, Nz, N, N)

    kernel_col = np.array([[[-1,1]]]).astype('float32')
    kernel_col = torch.as_tensor(np.reshape(kernel_col, (1,1)+kernel_col.shape)).cuda()

    kernel_row = np.array([[[-1],[1]]]).astype('float32')
    kernel_row = torch.as_tensor(np.reshape(kernel_row, (1,1)+kernel_row.shape)).cuda()

    kernel_slice = np.array([[[-1]],[[1]]]).astype('float32')
    kernel_slice = torch.as_tensor(np.reshape(kernel_slice, (1,1)+kernel_slice.shape)).cuda()

    # The intensity differences across rows (Upwind / Forward)
    # (Transpose to switch M and Nz)
    D_img[:, 0, :, :-1, :] = torch.transpose(torch.nn.functional.conv3d(img_tensor, kernel_row, bias=None, stride=1, padding = 0)[:, 0, :, :, :], 1, 0)

    # The intensity differences across columns (Upwind / Forward)
    # (Transpose to switch M and Nz)
    D_img[:, 1, :, :, :-1] = torch.transpose(torch.nn.functional.conv3d(img_tensor, kernel_col, bias=None, stride=1, padding = 0)[:, 0, :, :, :], 1, 0)

    # The intensity differences across slices (Upwind / Forward)
    # (Transpose to switch M and Nz)
    i_d = 2
    if Nz > 1 and reg_z_over_reg > 0:
        D_img[:-1, i_d, :, :, :] = np.sqrt(reg_z_over_reg) * torch.transpose(torch.nn.functional.conv3d(img_tensor, kernel_slice, bias=None, stride=1, padding = 0)[:, 0, :, :, :], 1, 0)
        i_d += 1

    # Reshape from (M, 1, Nz, N, N) to (Nz, M, N, N)
    img_tensor = torch.transpose(img_tensor, 0, 2)[:,0,:,:,:]

    # The intensity differences across times (Upwind / Forward)
    # Given that M is usually <10, it's not worth using the convolution operator there
    if reg_time > 0 and M > 1:

        # D_img[:, i_d, :-1, :, :] = np.sqrt(reg_time) * torch.reshape(torch.as_tensor((img[:, 1:, :, :] - img[:, :-1, :, :])), [Nz, M-1, N, N])
        D_img[:, i_d, :-1, :, :] = np.sqrt(reg_time) * (img_tensor[:, 1:, :, :] - img_tensor[:, :-1, :, :])
        # D_img[:, i_d, :-1, :, :] = np.sqrt(reg_time) * torch.nn.functional.conv3d(img_tensor, kernel_slice, bias=None, stride=1, padding = 0)[:, 0, :, :, :]

        if isinstance(mask_static, np.ndarray):
            mask_static = np.tile(mask_static, [Nz, M, 1, 1])
            D_img_temp = D_img[:,i_d,:,:,:].copy()
            D_img_temp[mask_static] *= np.sqrt(factor_reg_static)
            D_img[:,i_d,:,:,:] = D_img_temp
        i_d += 1

    del img_tensor, kernel_row, kernel_col, kernel_slice

    if not return_pytorch_tensor:
        D_img2 = D_img.cpu().detach().numpy()
        del D_img
        D_img = D_img2

    return D_img

def D_central(img, reg_z_over_reg = 1.0, reg_time = 0, mask_static = False, factor_reg_static = 0, return_pytorch_tensor = False):
    '''
    Calculates the output of the input image img by the operator D (gradient discretized using central scheme)

    Parameters
    ----------
    img : np.ndarray
        The array of the input image data of dimensions Nz x M x N x N.
    reg_z_over_reg : float
        The ratio of the regularization parameter in the z direction, versus the x-y plane.
    reg_time : float
        The ratio (\mu) of the regularization parameter in the time direction, versus the x-y plane.
    mask_static : np.ndarray
        An array of dimensions 1 x 1 x N x N serving as a mask to indicate pixels on which to enforce a different
        time regularization parameter, for instance used to enforce more static regions in the image.
    factor_reg_static : float
        The regularization parameter to compute in the region of the image specified by mask_static.
    return_pytorch_tensor : boolean
        Whether to return a numpy np.ndarray or a PyTorch torch.Tensor

    Returns
    -------
    np.ndarray or torch.Tensor
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
    i_d = 2

    D_img = torch.zeros([Nz, N_d, M, N, N]).cuda()

    # Reshape input as (M, 1, Nz, N, N), such that can use conv3d directly on Nz (typically Nz >> M).
    img_tensor = np.reshape(img, (1,)+img.shape).astype('float32') # (1, Nz, M, N, N)
    img_tensor = torch.as_tensor(np.transpose(img_tensor.astype('float32'), [2, 0, 1, 3, 4])).cuda() # (M, 1, Nz, N, N)

    kernel_col = np.array([[[-1,0,1]]]).astype('float32')
    kernel_col = torch.as_tensor(np.reshape(kernel_col, (1,1)+kernel_col.shape)).cuda()

    kernel_row = np.array([[[-1],[0], [1]]]).astype('float32')
    kernel_row = torch.as_tensor(np.reshape(kernel_row, (1,1)+kernel_row.shape)).cuda()

    kernel_slice = np.array([[[-1]], [[0]],[[1]]]).astype('float32')
    kernel_slice = torch.as_tensor(np.reshape(kernel_slice, (1,1)+kernel_slice.shape)).cuda()

    # The intensity differences across rows (Upwind / Forward)
    # (Transpose to switch M and Nz)
    D_img[:, 0, :, 1:-1, :] = torch.transpose(torch.nn.functional.conv3d(img_tensor, kernel_row, bias=None, stride=1, padding = 0)[:, 0, :, :, :], 1, 0)

    # The intensity differences across columns (Upwind / Forward)
    # (Transpose to switch M and Nz)
    D_img[:, 1, :, :, 1:-1] = torch.transpose(torch.nn.functional.conv3d(img_tensor, kernel_col, bias=None, stride=1, padding = 0)[:, 0, :, :, :], 1, 0)

    # The intensity differences across slices (Upwind / Forward)
    # (Transpose to switch M and Nz)
    i_d = 2
    if Nz > 1 and reg_z_over_reg > 0:
        if Nz == 2: # Use upwind scheme instead
            D_img[:-1, i_d, :, :, :] = np.sqrt(reg_z_over_reg) * (img_tensor[1:, :, :, :] - img_tensor[:-1, :, :, :])
        else:
            D_img[1:-1, i_d, :, :, :] = np.sqrt(reg_z_over_reg) * torch.transpose(torch.nn.functional.conv3d(img_tensor, kernel_slice, bias=None, stride=1, padding = 0)[:, 0, :, :, :], 1, 0)
        i_d += 1

    # Reshape from (M, 1, Nz, N, N) to (Nz, M, N, N)
    img_tensor = torch.transpose(img_tensor, 0, 2)[:,0,:,:,:]

    # The intensity differences across times (Upwind / Forward)
    # Given that M is usually <10, it's not worth using the convolution operator there
    if reg_time > 0 and M > 1:
        if M == 2: # Use upwind scheme instead
            D_img[:, i_d, :-1, :, :] = np.sqrt(reg_time) * (img_tensor[:, 1:, :, :] - img_tensor[:, :-1, :, :])
        else:
            D_img[:, i_d, 1:-1, :, :] = np.sqrt(reg_time) * (img_tensor[:, 2:, :, :] - img_tensor[:, :-2, :, :])

        if isinstance(mask_static, np.ndarray):
            mask_static = np.tile(mask_static, [Nz, M, 1, 1])
            D_img_temp = D_img[:,i_d,:,:,:].copy()
            D_img_temp[mask_static] *= np.sqrt(factor_reg_static)
            D_img[:,i_d,:,:,:] = D_img_temp
        i_d += 1

    del img_tensor, kernel_row, kernel_col, kernel_slice

    if not return_pytorch_tensor:
        D_img2 = D_img.cpu().detach().numpy()
        del D_img
        D_img = D_img2

    return D_img / 2.0

def D_T_hybrid(img, reg_z_over_reg = 1.0, reg_time = 0, mask_static = False, factor_reg_static = 0, return_pytorch_tensor = False):
    '''
    Calculates the output of the input image img by the operator D^T (tranposed gradient discretized using hybrid scheme)

    Parameters
    ----------
    img : np.ndarray or torch.Tensor
        The array of the input image data of dimensions Nz x Nd x M x N x N.
    reg_z_over_reg : float
        The ratio of the regularization parameter in the z direction, versus the x-y plane.
    reg_time : float
        The ratio (\mu) of the regularization parameter in the time direction, versus the x-y plane.
    mask_static : np.ndarray
        An of dimensions 1 x 1 x N x N serving as a mask to indicate pixels on which to enforce a different
        time regularization parameter, for instance used to enforce more static regions in the image.
    factor_reg_static : float
        The regularization parameter to compute in the region of the image specified by mask_static.
    return_pytorch_tensor : boolean
        Whether to return a numpy np.ndarray or a PyTorch torch.Tensor

    Returns
    -------
    np.ndarray or torch.Tensor
        The array of the discretized gradient D^T(img) of dimensions Nz x M x N x N.
    '''

    if reg_z_over_reg == np.nan:
        reg_z_over_reg = 0.0

    Nz = img.shape[0]
    N_d = img.shape[1]
    M = img.shape[2]
    N = img.shape[-1]

    D_T_img = torch.zeros([Nz, M, N, N]).cuda()

    if type(img) != torch.Tensor:
        img = torch.as_tensor(img.astype('float32')).cuda()
    else:
        img = img.cuda()

    kernel_col = np.array([[[1,-1]]]).astype('float32')
    kernel_col = torch.as_tensor(np.reshape(kernel_col, (1,1)+kernel_col.shape)).cuda()

    kernel_row = np.array([[[1],[-1]]]).astype('float32')
    kernel_row = torch.as_tensor(np.reshape(kernel_row, (1,1)+kernel_row.shape)).cuda()

    kernel_slice = np.array([[[1]],[[-1]]]).astype('float32')
    kernel_slice = torch.as_tensor(np.reshape(kernel_slice, (1,1)+kernel_slice.shape)).cuda()

    # Forward row term
    D_T_img[:,:,1:-1,:] += torch.nn.functional.conv3d(img[:,0:1,:,:-1,:], kernel_row, bias=None, stride=1, padding = 0)[:,0,:,:,:]
    D_T_img[:,:,0,:] += -img[:,0,:,0,:]
    D_T_img[:,:,-1,:] += img[:,0,:,-2,:]

    # Forward col term
    D_T_img[:,:,:,1:-1] += torch.nn.functional.conv3d(img[:,1:2,:,:,:-1], kernel_col, bias=None, stride=1, padding = 0)[:,0,:,:,:]
    D_T_img[:,:,:,0] += -img[:,1,:,:,0]
    D_T_img[:,:,:,-1] += img[:,1,:,:,-2]

    # Backward row term
    D_T_img[:,:,1:-1,:] += torch.nn.functional.conv3d(img[:,2:3,:,1:,:], kernel_row, bias=None, stride=1, padding = 0)[:,0,:,:,:]
    D_T_img[:,:,0,:] += -img[:,2,:,1,:]
    D_T_img[:,:,-1,:] += img[:,2,:,-1,:]

    # Backward col term
    D_T_img[:,:,:,1:-1] += torch.nn.functional.conv3d(img[:,3:4,:,:,1:], kernel_col, bias=None, stride=1, padding = 0)[:,0,:,:,:]
    D_T_img[:,:,:,0] += -img[:,3,:,:,1]
    D_T_img[:,:,:,-1] += img[:,3,:,:,-1]

    # From (Nz, Nd, M, N, N) to (M, Nd, Nz, N, N)
    img = torch.transpose(img, 0, 2)

    i_d = 4
    if Nz > 1 and reg_z_over_reg > 0:

        # Forward slices term
        D_T_img[0,:,:,:] += -np.sqrt(reg_z_over_reg) * img[:,i_d,0,:,:]
        D_T_img[-1,:,:,:] += np.sqrt(reg_z_over_reg) * img[:,i_d,-2,:,:]

        D_T_img[1:-1,:,:,:] += np.sqrt(reg_z_over_reg) * torch.transpose(torch.nn.functional.conv3d(img[:,i_d:i_d+1,:-1,:,:], kernel_slice, bias=None, stride=1, padding = 0), 0, 2)[:,0,:,:,:]
        # Equivalent to above convolution, but higher computational cost
        # D_T_img[1:-1,:,:,:] += np.sqrt(reg_z_over_reg) * torch.transpose(img[:,i_d,:-2,:,:]-img[:,i_d,1:-1,:,:], 0, 1)
        i_d += 1

        # Backward slices term
        D_T_img[0,:,:,:] += -np.sqrt(reg_z_over_reg) * img[:,i_d,1,:,:]
        D_T_img[-1,:,:,:] += np.sqrt(reg_z_over_reg) * img[:,i_d,-1,:,:]

        D_T_img[1:-1,:,:,:] += np.sqrt(reg_z_over_reg) * torch.transpose(torch.nn.functional.conv3d(img[:,i_d:i_d+1,1:,:,:], kernel_slice, bias=None, stride=1, padding = 0), 0, 2)[:,0,:,:,:]
        # Equivalent to above convolution, but higher computational cost
        # D_T_img[1:-1,:,:,:] += np.sqrt(reg_z_over_reg) * torch.transpose(img[:,i_d,1:-1,:,:]-img[:,i_d,2:,:,:], 0, 1)
        i_d += 1

    # From (M, Nd, Nz, N, N) to (Nz, Nd, M, N, N)
    img = torch.transpose(img, 0, 2)

    # Given that M is usually <10, it's not worth using the convolution operator there
    if reg_time > 0 and M > 1:
        D_T_img_time_update = torch.zeros_like(D_T_img)

        # Forward time term
        D_T_img_time_update[:,1:-1,:,:] += np.sqrt(reg_time) * (img[:,i_d,:-2,:,:]-img[:,i_d,1:-1,:,:])
        D_T_img_time_update[:,0,:,:] += -np.sqrt(reg_time) * img[:,i_d,0,:,:]
        D_T_img_time_update[:,-1,:,:] += np.sqrt(reg_time) * img[:,i_d,-2,:,:]
        i_d += 1

        # Backward time term
        D_T_img_time_update[:,1:-1,:,:] += np.sqrt(reg_time) * (img[:,i_d,1:-1,:,:]-img[:,i_d,2:,:,:])
        D_T_img_time_update[:,0,:,:] += -np.sqrt(reg_time) * img[:,i_d,1,:,:]
        D_T_img_time_update[:,-1,:,:] += np.sqrt(reg_time) * img[:,i_d,-1,:,:]
        i_d += 1

        if isinstance(mask_static, np.ndarray):
            mask_static = np.tile(mask_static, [Nz, M, 1, 1])
            D_T_img_time_update[mask_static] *= np.sqrt(factor_reg_static)
        D_T_img += D_T_img_time_update
        i_d += 1

    del img, kernel_row, kernel_col, kernel_slice

    if not return_pytorch_tensor:
        D_T_img2 = D_T_img.cpu().detach().numpy()
        del D_T_img
        D_T_img = D_T_img2

    return(D_T_img/np.sqrt(2.0))

def D_T_downwind(img, reg_z_over_reg = 1.0, reg_time = 0, mask_static = False, factor_reg_static = 0, return_pytorch_tensor = False):
    '''
    Calculates the output of the input image img by the operator D^T (tranposed gradient discretized using downwind scheme)

    Parameters
    ----------
    img : np.ndarray or torch.Tensor
        The array of the input image data of dimensions Nz x Nd x M x N x N.
    reg_z_over_reg : float
        The ratio of the regularization parameter in the z direction, versus the x-y plane.
    reg_time : float
        The ratio (\mu) of the regularization parameter in the time direction, versus the x-y plane.
    mask_static : np.ndarray
        An of dimensions 1 x 1 x N x N serving as a mask to indicate pixels on which to enforce a different
        time regularization parameter, for instance used to enforce more static regions in the image.
    factor_reg_static : float
        The regularization parameter to compute in the region of the image specified by mask_static.
    return_pytorch_tensor : boolean
        Whether to return a numpy np.ndarray or a PyTorch torch.Tensor

    Returns
    -------
    np.ndarray or torch.Tensor
        The array of the discretized gradient D^T(img) of dimensions Nz x M x N x N.
    '''

    if reg_z_over_reg == np.nan:
        reg_z_over_reg = 0.0

    Nz = img.shape[0]
    N_d = img.shape[1]
    M = img.shape[2]
    N = img.shape[-1]

    D_T_img = torch.zeros([Nz, M, N, N]).cuda()

    if type(img) != torch.Tensor:
        img = torch.as_tensor(img.astype('float32')).cuda()
    else:
        img = img.cuda()

    kernel_col = np.array([[[1,-1]]]).astype('float32')
    kernel_col = torch.as_tensor(np.reshape(kernel_col, (1,1)+kernel_col.shape)).cuda()

    kernel_row = np.array([[[1],[-1]]]).astype('float32')
    kernel_row = torch.as_tensor(np.reshape(kernel_row, (1,1)+kernel_row.shape)).cuda()

    kernel_slice = np.array([[[1]],[[-1]]]).astype('float32')
    kernel_slice = torch.as_tensor(np.reshape(kernel_slice, (1,1)+kernel_slice.shape)).cuda()

    # Backward row term
    D_T_img[:,:,1:-1,:] += torch.nn.functional.conv3d(img[:,0:1,:,1:,:], kernel_row, bias=None, stride=1, padding = 0)[:,0,:,:,:]
    D_T_img[:,:,0,:] += -img[:,0,:,1,:]
    D_T_img[:,:,-1,:] += img[:,0,:,-1,:]

    # Backward col term
    D_T_img[:,:,:,1:-1] += torch.nn.functional.conv3d(img[:,1:2,:,:,1:], kernel_col, bias=None, stride=1, padding = 0)[:,0,:,:,:]
    D_T_img[:,:,:,0] += -img[:,1,:,:,1]
    D_T_img[:,:,:,-1] += img[:,1,:,:,-1]

    # From (Nz, Nd, M, N, N) to (M, Nd, Nz, N, N)
    img = torch.transpose(img, 0, 2)

    i_d = 2
    if Nz > 1 and reg_z_over_reg > 0:
        # Backward slices term
        D_T_img[0,:,:,:] += -np.sqrt(reg_z_over_reg) * img[:,i_d,1,:,:]
        D_T_img[-1,:,:,:] += np.sqrt(reg_z_over_reg) * img[:,i_d,-1,:,:]

        D_T_img[1:-1,:,:,:] += np.sqrt(reg_z_over_reg) * torch.transpose(torch.nn.functional.conv3d(img[:,i_d:i_d+1,1:,:,:], kernel_slice, bias=None, stride=1, padding = 0), 0, 2)[:,0,:,:,:]
        # Equivalent to above convolution, but higher computational cost
        # D_T_img[1:-1,:,:,:] += np.sqrt(reg_z_over_reg) * torch.transpose(img[:,i_d,1:-1,:,:]-img[:,i_d,2:,:,:], 0, 1)
        i_d += 1

    # From (M, Nd, Nz, N, N) to (Nz, Nd, M, N, N)
    img = torch.transpose(img, 0, 2)

    if reg_time > 0 and M > 1:
        # Backward time term
        # Given that M is usually <10, it's not worth using the convolution operator there
        D_T_img_time_update = torch.zeros_like(D_T_img)

        D_T_img_time_update[:,1:-1,:,:] += np.sqrt(reg_time) * (img[:,i_d,1:-1,:,:]-img[:,i_d,2:,:,:])
        D_T_img_time_update[:,0,:,:] += -np.sqrt(reg_time) * img[:,i_d,1,:,:]
        D_T_img_time_update[:,-1,:,:] += np.sqrt(reg_time) * img[:,i_d,-1,:,:]

        if isinstance(mask_static, np.ndarray):
            mask_static = np.tile(mask_static, [Nz, M, 1, 1])
            D_T_img_time_update[mask_static] *= np.sqrt(factor_reg_static)
        D_T_img += D_T_img_time_update
        i_d += 1

    del img, kernel_row, kernel_col, kernel_slice

    if not return_pytorch_tensor:
        D_T_img2 = D_T_img.cpu().detach().numpy()
        del D_T_img
        D_T_img = D_T_img2

    return(D_T_img)

def D_T_upwind(img, reg_z_over_reg = 1.0, reg_time = 0, mask_static = False, factor_reg_static = 0, return_pytorch_tensor = False):
    '''
    Calculates the output of the input image img by the operator D^T (tranposed gradient discretized using upwind scheme)

    Parameters
    ----------
    img : np.ndarray or torch.Tensor
        The array of the input image data of dimensions Nz x Nd x M x N x N.
    reg_z_over_reg : float
        The ratio of the regularization parameter in the z direction, versus the x-y plane.
    reg_time : float
        The ratio (\mu) of the regularization parameter in the time direction, versus the x-y plane.
    mask_static : np.ndarray
        An of dimensions 1 x 1 x N x N serving as a mask to indicate pixels on which to enforce a different
        time regularization parameter, for instance used to enforce more static regions in the image.
    factor_reg_static : float
        The regularization parameter to compute in the region of the image specified by mask_static.
    return_pytorch_tensor : boolean
        Whether to return a numpy np.ndarray or a PyTorch torch.Tensor

    Returns
    -------
    np.ndarray or torch.Tensor
        The array of the discretized gradient D^T(img) of dimensions Nz x M x N x N.
    '''

    if reg_z_over_reg == np.nan:
        reg_z_over_reg = 0.0

    Nz = img.shape[0]
    N_d = img.shape[1]
    M = img.shape[2]
    N = img.shape[-1]

    D_T_img = torch.zeros([Nz, M, N, N]).cuda()

    if type(img) != torch.Tensor:
        img = torch.as_tensor(img.astype('float32')).cuda()
    else:
        img = img.cuda()

    kernel_col = np.array([[[1,-1]]]).astype('float32')
    kernel_col = torch.as_tensor(np.reshape(kernel_col, (1,1)+kernel_col.shape)).cuda()

    kernel_row = np.array([[[1],[-1]]]).astype('float32')
    kernel_row = torch.as_tensor(np.reshape(kernel_row, (1,1)+kernel_row.shape)).cuda()

    kernel_slice = np.array([[[1]],[[-1]]]).astype('float32')
    kernel_slice = torch.as_tensor(np.reshape(kernel_slice, (1,1)+kernel_slice.shape)).cuda()

    # Forward row term
    D_T_img[:,:,1:-1,:] += torch.nn.functional.conv3d(img[:,0:1,:,:-1,:], kernel_row, bias=None, stride=1, padding = 0)[:,0,:,:,:]
    D_T_img[:,:,0,:] += -img[:,0,:,0,:]
    D_T_img[:,:,-1,:] += img[:,0,:,-2,:]

    # Forward col term
    D_T_img[:,:,:,1:-1] += torch.nn.functional.conv3d(img[:,1:2,:,:,:-1], kernel_col, bias=None, stride=1, padding = 0)[:,0,:,:,:]
    D_T_img[:,:,:,0] += -img[:,1,:,:,0]
    D_T_img[:,:,:,-1] += img[:,1,:,:,-2]

    # From (Nz, Nd, M, N, N) to (M, Nd, Nz, N, N)
    img = torch.transpose(img, 0, 2)

    i_d = 2
    if Nz > 1 and reg_z_over_reg > 0:
        # Forward slices term
        D_T_img[0,:,:,:] += -np.sqrt(reg_z_over_reg) * img[:,i_d,0,:,:]
        D_T_img[-1,:,:,:] += np.sqrt(reg_z_over_reg) * img[:,i_d,-2,:,:]

        D_T_img[1:-1,:,:,:] += np.sqrt(reg_z_over_reg) * torch.transpose(torch.nn.functional.conv3d(img[:,i_d:i_d+1,:-1,:,:], kernel_slice, bias=None, stride=1, padding = 0), 0, 2)[:,0,:,:,:]
        # Equivalent to above convolution, but higher computational cost
        # D_T_img[1:-1,:,:,:] += np.sqrt(reg_z_over_reg) * torch.transpose(img[:,i_d,:-2,:,:]-img[:,i_d,1:-1,:,:], 0, 1)

        i_d += 1

    # From (M, Nd, Nz, N, N) to (Nz, Nd, M, N, N)
    img = torch.transpose(img, 0, 2)

    if reg_time > 0 and M > 1:
        # Forward time term
        # Given that M is usually <10, it's not worth using the convolution operator there
        D_T_img_time_update = torch.zeros_like(D_T_img)

        D_T_img_time_update[:,1:-1,:,:] += np.sqrt(reg_time) * (img[:,i_d,:-2,:,:]-img[:,i_d,1:-1,:,:])
        D_T_img_time_update[:,0,:,:] += -np.sqrt(reg_time) * img[:,i_d,0,:,:]
        D_T_img_time_update[:,-1,:,:] += np.sqrt(reg_time) * img[:,i_d,-2,:,:]

        if isinstance(mask_static, np.ndarray):
            mask_static = np.tile(mask_static, [Nz, M, 1, 1])
            D_T_img_time_update[mask_static] *= np.sqrt(factor_reg_static)
        D_T_img += D_T_img_time_update
        i_d += 1

    del img, kernel_row, kernel_col, kernel_slice

    if not return_pytorch_tensor:
        D_T_img2 = D_T_img.cpu().detach().numpy()
        del D_T_img
        D_T_img = D_T_img2

    return(D_T_img)

def D_T_central(img, reg_z_over_reg = 1.0, reg_time = 0, mask_static = False, factor_reg_static = 0, return_pytorch_tensor = False):
    '''
    Calculates the output of the input image img by the operator D^T (tranposed gradient discretized using central scheme)

    Parameters
    ----------
    img : np.ndarray or torch.Tensor
        The array of the input image data of dimensions Nz x Nd x M x N x N.
    reg_z_over_reg : float
        The ratio of the regularization parameter in the z direction, versus the x-y plane.
    reg_time : float
        The ratio (\mu) of the regularization parameter in the time direction, versus the x-y plane.
    mask_static : np.ndarray
        An of dimensions 1 x 1 x N x N serving as a mask to indicate pixels on which to enforce a different
        time regularization parameter, for instance used to enforce more static regions in the image.
    factor_reg_static : float
        The regularization parameter to compute in the region of the image specified by mask_static.
    return_pytorch_tensor : boolean
        Whether to return a numpy np.ndarray or a PyTorch torch.Tensor

    Returns
    -------
    np.ndarray or torch.Tensor
        The array of the discretized gradient D^T(img) of dimensions Nz x M x N x N.
    '''

    if reg_z_over_reg == np.nan:
        reg_z_over_reg = 0.0

    Nz = img.shape[0]
    N_d = img.shape[1]
    M = img.shape[2]
    N = img.shape[-1]

    D_T_img = torch.zeros([Nz, M, N, N]).cuda()

    if type(img) != torch.Tensor:
        img = torch.as_tensor(img.astype('float32')).cuda()
    else:
        img = img.cuda()


    kernel_col = np.array([[[1,0,-1]]]).astype('float32')
    kernel_col = torch.as_tensor(np.reshape(kernel_col, (1,1)+kernel_col.shape)).cuda()

    kernel_row = np.array([[[1],[0], [-1]]]).astype('float32')
    kernel_row = torch.as_tensor(np.reshape(kernel_row, (1,1)+kernel_row.shape)).cuda()

    kernel_slice = np.array([[[1]],[[0]],[[-1]]]).astype('float32')
    kernel_slice = torch.as_tensor(np.reshape(kernel_slice, (1,1)+kernel_slice.shape)).cuda()

    # Forward row term
    D_T_img[:,:,2:-2,:] += torch.nn.functional.conv3d(img[:,0:1,:,1:-1,:], kernel_row, bias=None, stride=1, padding = 0)[:,0,:,:,:]
    D_T_img[:,:,0:2,:] += -img[:,0,:,1:3,:]
    D_T_img[:,:,-2:,:] += img[:,0,:,-3:-1,:]

    # Forward col term
    D_T_img[:,:,:,2:-2] += torch.nn.functional.conv3d(img[:,1:2,:,:,1:-1], kernel_col, bias=None, stride=1, padding = 0)[:,0,:,:,:]
    D_T_img[:,:,:,0:2] += -img[:,1,:,:,1:3]
    D_T_img[:,:,:,-2:] += img[:,1,:,:,-3:-1]

    # From (Nz, Nd, M, N, N) to (M, Nd, Nz, N, N)
    img = torch.transpose(img, 0, 2)

    i_d = 2
    # Forward slices term
    if Nz > 1 and reg_z_over_reg > 0:
        if Nz == 2: # Use upwind scheme instead
            D_T_img[1:,:,:,:] += np.sqrt(reg_z_over_reg) * img[:-1, i_d, :, :, :]
            D_T_img[:-1,:,:,:] += -np.sqrt(reg_z_over_reg) * img[:-1, i_d, :, :, :]
        else:
            D_T_img[0:2,:,:,:] += - np.sqrt(reg_z_over_reg) * torch.transpose(img[:,i_d,1:3,:,:], 0, 1)
            D_T_img[-2:,:,:,:] += np.sqrt(reg_z_over_reg) * torch.transpose(img[:,i_d,-3:-1,:,:], 0, 1)
            D_T_img[2:-2,:,:,:] += np.sqrt(reg_z_over_reg) * torch.transpose(torch.nn.functional.conv3d(img[:,i_d:i_d+1,1:-1,:,:], kernel_slice, bias=None, stride=1, padding = 0), 0, 2)[:,0,:,:,:]
        i_d += 1

    # From (M, Nd, Nz, N, N) to (Nz, Nd, M, N, N)
    img = torch.transpose(img, 0, 2)

    if reg_time > 0 and M > 1:
        # Forward time term
        # Given that M is usually <10, it's not worth using the convolution operator there
        D_T_img_time_update = torch.zeros_like(D_T_img)

        if M == 2:
            D_T_img_time_update[:, 1:, :, :] += np.sqrt(reg_time) * img[:, i_d, :-1, :, :]
            D_T_img_time_update[:, :-1, :, :] += -np.sqrt(reg_time) * img[:, i_d, :-1, :, :]
        else:
            D_T_img_time_update[:, 2:, :, :] += np.sqrt(reg_time) * img[:, i_d, 1:-1, :, :]
            D_T_img_time_update[:, :-2, :, :] += -np.sqrt(reg_time) * img[:, i_d, 1:-1, :, :]

        if isinstance(mask_static, np.ndarray):
            mask_static = np.tile(mask_static, [Nz, M, 1, 1])
            D_T_img_time_update[mask_static] *= np.sqrt(factor_reg_static)
        D_T_img += D_T_img_time_update
        i_d += 1

    del img, kernel_row, kernel_col, kernel_slice

    D_T_img /= 2.0

    if not return_pytorch_tensor:
        D_T_img2 = D_T_img.cpu().detach().numpy()
        del D_T_img
        D_T_img = D_T_img2

    return(D_T_img)
