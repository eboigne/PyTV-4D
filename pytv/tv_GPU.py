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
import torch
import pytv.tv_2d_GPU

def tv_hybrid(img, mask = [], reg_z_over_reg = 1.0, match_2D_form = False):
    '''
    Calculates the total variation and a subgradient of the input image img using the hybrid gradient discretization

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
    G_cpu : np.ndarray
        A sub-gradient array of the total variation term of same dimensions as the input image img.
    '''

    if mask != []:
        img[~mask] = 0

    if len(img.shape) == 2:
        return(pytv.tv_2d_GPU.tv_hybrid(img))
    elif (len(img.shape) == 3 and img.shape[0] < 3):
        return(pytv.tv_2d_GPU.tv_hybrid(img[0]))

    kernel_col = np.array([[[-1,1]]]).astype('float32')
    kernel_col = torch.as_tensor(np.reshape(kernel_col, (1,1)+kernel_col.shape)).cuda()

    kernel_row = np.array([[[-1],[1]]]).astype('float32')
    kernel_row = torch.as_tensor(np.reshape(kernel_row, (1,1)+kernel_row.shape)).cuda()

    kernel_slice = np.array([[[-1]], [[1]]]).astype('float32')
    kernel_slice = torch.as_tensor(np.reshape(kernel_slice, (1,1)+kernel_slice.shape)).cuda()

    img = np.reshape(img, (1,1)+img.shape).astype('float32')
    img_tensor = torch.as_tensor(img).cuda()
    row_diff_tensor = (torch.zeros_like(img_tensor)).squeeze()
    col_diff_tensor = (torch.zeros_like(img_tensor)).squeeze()
    slice_diff_tensor = (torch.zeros_like(img_tensor)).squeeze()

    row_diff_tensor[:, :-1, :] = torch.nn.functional.conv3d(img_tensor, kernel_row, bias=None, stride=1, padding = 0).squeeze()
    col_diff_tensor[:, :, :-1] = torch.nn.functional.conv3d(img_tensor, kernel_col, bias=None, stride=1, padding = 0).squeeze()
    slice_diff_tensor[:-1, :, :] = np.sqrt(reg_z_over_reg) * torch.nn.functional.conv3d(img_tensor, kernel_slice, bias=None, stride=1, padding = 0).squeeze()

    # To match CPU explicit versions
    row_diff_tensor[:, :, -1] = 0
    row_diff_tensor[-1, :, :] = 0
    col_diff_tensor[:, -1, :] = 0
    col_diff_tensor[-1, :, :] = 0
    slice_diff_tensor[:, -1, :] = 0
    slice_diff_tensor[:, :, -1] = 0

    grad_norms = (torch.zeros_like(img_tensor)).squeeze()
    if match_2D_form:
        grad_norms[:-1, :-1, :-1] = torch.sqrt(torch.square(row_diff_tensor[:-1, :-1, 1:]) + torch.square(row_diff_tensor[:-1, :-1, :-1])
                                               + torch.square(col_diff_tensor[:-1, 1:, :-1]) + torch.square(col_diff_tensor[:-1, :-1, :-1])
                                               + torch.square(slice_diff_tensor[:-1, 1:, 1:]) + torch.square(slice_diff_tensor[:-1, :-1, :-1])) / np.sqrt(2)
    else:
        grad_norms[:-1, :-1, :-1] = torch.sqrt(torch.square(row_diff_tensor[1:, :-1, 1:]) + torch.square(row_diff_tensor[:-1, :-1, :-1])
                                               + torch.square(col_diff_tensor[1:, 1:, :-1]) + torch.square(col_diff_tensor[:-1, :-1, :-1])
                                               + torch.square(slice_diff_tensor[:-1, 1:, 1:]) + torch.square(slice_diff_tensor[:-1, :-1, :-1])) / np.sqrt(2)
    tv = grad_norms.sum().cpu().detach().numpy().squeeze()
    grad_norms[grad_norms == 0] = np.inf

    G = torch.zeros_like(img_tensor).squeeze()

    G[:-1, :-1, :-1] = - (row_diff_tensor+col_diff_tensor+slice_diff_tensor)[:-1, :-1, :-1]/grad_norms[:-1, :-1, :-1]

    G[:-1, :-1, 1:] += col_diff_tensor[:-1, :-1, :-1]/grad_norms[:-1, :-1, :-1]
    G[:-1, 1:, :-1] += row_diff_tensor[:-1, :-1, :-1]/grad_norms[:-1, :-1, :-1]
    G[1:, :-1, :-1] += slice_diff_tensor[:-1, :-1, :-1]/grad_norms[:-1, :-1, :-1]

    G[1:, 1:, :-1] += -col_diff_tensor[1:, 1:, :-1]/grad_norms[:-1, :-1, :-1]
    G[1:, :-1, 1:] += -row_diff_tensor[1:, :-1, 1:]/grad_norms[:-1, :-1, :-1]
    G[:-1, 1:, 1:] += -slice_diff_tensor[:-1, 1:, 1:]/grad_norms[:-1, :-1, :-1]

    G[1:, 1:, 1:] += (row_diff_tensor[1:, :-1, 1:] + col_diff_tensor[1:, 1:, :-1] + slice_diff_tensor[:-1, 1:, 1:])/grad_norms[:-1, :-1, :-1]

    G_cpu = G.cpu().detach().numpy().squeeze()
    torch.cuda.empty_cache()
    del G, grad_norms, row_diff_tensor, col_diff_tensor, slice_diff_tensor, img_tensor, kernel_row, kernel_col, kernel_slice

    return(tv, G_cpu)

def tv_downwind(img, mask = [], reg_z_over_reg = 1.0):
    '''
    Calculates the total variation and a subgradient of the input image img using the downwind gradient discretization

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
    G_cpu : np.ndarray
        A sub-gradient array of the total variation term of same dimensions as the input image img.
    '''

    if mask != []:
        img[~mask] = 0

    if len(img.shape) == 2:
        return(pytv.tv_2d_GPU.tv_downwind(img))
    elif (len(img.shape) == 3 and img.shape[0] < 3):
        return(pytv.tv_2d_GPU.tv_downwind(img[0]))

    kernel_col = np.array([[[-1,1]]]).astype('float32')
    kernel_col = torch.as_tensor(np.reshape(kernel_col, (1,1)+kernel_col.shape)).cuda()

    kernel_row = np.array([[[-1],[1]]]).astype('float32')
    kernel_row = torch.as_tensor(np.reshape(kernel_row, (1,1)+kernel_row.shape)).cuda()

    kernel_slice = np.array([[[-1]], [[1]]]).astype('float32')
    kernel_slice = torch.as_tensor(np.reshape(kernel_slice, (1,1)+kernel_slice.shape)).cuda()

    img = np.reshape(img, (1,1)+img.shape).astype('float32')
    img_tensor = torch.as_tensor(img).cuda()
    row_diff_tensor = (torch.zeros_like(img_tensor)).squeeze()
    col_diff_tensor = (torch.zeros_like(img_tensor)).squeeze()
    slice_diff_tensor = (torch.zeros_like(img_tensor)).squeeze()

    row_diff_tensor[:, :-1, :] = torch.nn.functional.conv3d(img_tensor, kernel_row, bias=None, stride=1, padding = 0).squeeze()
    col_diff_tensor[:, :, :-1] = torch.nn.functional.conv3d(img_tensor, kernel_col, bias=None, stride=1, padding = 0).squeeze()
    slice_diff_tensor[:-1, :, :] = np.sqrt(reg_z_over_reg) * torch.nn.functional.conv3d(img_tensor, kernel_slice, bias=None, stride=1, padding = 0).squeeze()

    # To match CPU explicit versions
    row_diff_tensor[:, :, -1] = 0
    row_diff_tensor[-1, :, :] = 0
    col_diff_tensor[:, -1, :] = 0
    col_diff_tensor[-1, :, :] = 0
    slice_diff_tensor[:, -1, :] = 0
    slice_diff_tensor[:, :, -1] = 0

    grad_norms = (torch.zeros_like(img_tensor)).squeeze()
    grad_norms[:-1, :-1, :-1] = torch.sqrt(torch.square(row_diff_tensor[1:, :-1, 1:])+torch.square(col_diff_tensor[1:, 1:, :-1])+torch.square(slice_diff_tensor[:-1, 1:, 1:]))
    tv = grad_norms.sum().cpu().detach().numpy().squeeze()
    grad_norms[grad_norms == 0] = np.inf
    
    G = torch.zeros_like(img_tensor).squeeze()
    G[1:, 1:, 1:] = (row_diff_tensor[1:, :-1, 1:] + col_diff_tensor[1:, 1:, :-1] + slice_diff_tensor[:-1, 1:, 1:]) / grad_norms[:-1, :-1, :-1]
    G[1:, :-1, 1:] += - row_diff_tensor[1:, :-1, 1:] / grad_norms[:-1, :-1, :-1]
    G[1:, 1:, :-1] += - col_diff_tensor[1:, 1:, :-1] / grad_norms[:-1, :-1, :-1]
    G[:-1, 1:, 1:] += - slice_diff_tensor[:-1, 1:, 1:] / grad_norms[:-1, :-1, :-1]

    G_cpu = G.cpu().detach().numpy().squeeze()
    torch.cuda.empty_cache()
    del G, grad_norms, row_diff_tensor, col_diff_tensor, slice_diff_tensor, img_tensor, kernel_row, kernel_col, kernel_slice
    
    return(tv, G_cpu)

def tv_upwind(img, mask = [], reg_z_over_reg = 1.0):
    '''
    Calculates the total variation and a subgradient of the input image img using the upwind gradient discretization

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
    G_cpu : np.ndarray
        A sub-gradient array of the total variation term of same dimensions as the input image img.
    '''

    if mask != []:
        img[~mask] = 0

    if len(img.shape) == 2:
        return(pytv.tv_2d_GPU.tv_upwind(img))
    elif (len(img.shape) == 3 and img.shape[0] < 3):
        return(pytv.tv_2d_GPU.tv_upwind(img[0]))

    kernel_col = np.array([[[-1,1]]]).astype('float32')
    kernel_col = torch.as_tensor(np.reshape(kernel_col, (1,1)+kernel_col.shape)).cuda()

    kernel_row = np.array([[[-1],[1]]]).astype('float32')
    kernel_row = torch.as_tensor(np.reshape(kernel_row, (1,1)+kernel_row.shape)).cuda()

    kernel_slice = np.array([[[-1]], [[1]]]).astype('float32')
    kernel_slice = torch.as_tensor(np.reshape(kernel_slice, (1,1)+kernel_slice.shape)).cuda()

    img = np.reshape(img, (1,1)+img.shape).astype('float32')
    img_tensor = torch.as_tensor(img).cuda()
    row_diff_tensor = (torch.zeros_like(img_tensor)).squeeze()
    col_diff_tensor = (torch.zeros_like(img_tensor)).squeeze()
    slice_diff_tensor = (torch.zeros_like(img_tensor)).squeeze()

    # img_tensor of size: (N,Cin,H,W), N: batch size, Cin: number of input channels.
    row_diff_tensor[:, :-1, :] = torch.nn.functional.conv3d(img_tensor, kernel_row, bias=None, stride=1, padding = 0).squeeze()
    col_diff_tensor[:, :, :-1] = torch.nn.functional.conv3d(img_tensor, kernel_col, bias=None, stride=1, padding = 0).squeeze()
    slice_diff_tensor[:-1, :, :] = np.sqrt(reg_z_over_reg) * torch.nn.functional.conv3d(img_tensor, kernel_slice, bias=None, stride=1, padding = 0).squeeze()

    # This is equivalent (without reshaping img as above), but slightly slower (15-20%)
    # row_diff_tensor[:, :-1, :] = img_tensor[:, 1:,:] - img_tensor[:,:-1,:]
    # col_diff_tensor[:, :, :-1] = img_tensor[:,:,1:] - img_tensor[:,:,:-1]
    # slice_diff_tensor[:-1, :, :] = img_tensor[1:,:,:] - img_tensor[:-1,:,:]

    # To match CPU explicit versions
    row_diff_tensor[:, :, -1] = 0
    row_diff_tensor[-1, :, :] = 0
    col_diff_tensor[:, -1, :] = 0
    col_diff_tensor[-1, :, :] = 0
    slice_diff_tensor[:, -1, :] = 0
    slice_diff_tensor[:, :, -1] = 0

    grad_norms = (torch.zeros_like(img_tensor)).squeeze()
    grad_norms = torch.sqrt(torch.square(row_diff_tensor)+torch.square(col_diff_tensor)+torch.square(slice_diff_tensor))
    tv = grad_norms.sum().cpu().detach().numpy().squeeze()
    grad_norms[grad_norms == 0] = np.inf
    
    G = torch.zeros_like(img_tensor).squeeze()
    G[:-1, :-1, :-1] =  - (row_diff_tensor+col_diff_tensor+slice_diff_tensor)[:-1, :-1, :-1]/grad_norms[:-1, :-1, :-1]
    G[:-1, 1:, :-1] += row_diff_tensor[:-1, :-1, :-1]/grad_norms[:-1, :-1, :-1]
    G[:-1, :-1, 1:] += col_diff_tensor[:-1, :-1, :-1]/grad_norms[:-1, :-1, :-1]
    G[1:, :-1, :-1] += slice_diff_tensor[:-1, :-1, :-1]/grad_norms[:-1, :-1, :-1]

    G_cpu = G.cpu().detach().numpy().squeeze()
    torch.cuda.empty_cache()
    del G, grad_norms, row_diff_tensor, col_diff_tensor, slice_diff_tensor, img_tensor, kernel_row, kernel_col, kernel_slice
    
    return(tv, G_cpu)

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
    G_cpu : np.ndarray
        A sub-gradient array of the total variation term of same dimensions as the input image img.
    '''

    if mask != []:
        img[~mask] = 0

    if len(img.shape) == 2:
        return(pytv.tv_2d_GPU.tv_centered(img))
    elif (len(img.shape) == 3 and img.shape[0] < 3):
        return(pytv.tv_2d_GPU.tv_centered(img[0]))

    kernel_row = np.array([[[-0.5], [0], [0.5]]]).astype('float32')
    kernel_row = torch.as_tensor(np.reshape(kernel_row, (1,1)+kernel_row.shape)).cuda()

    kernel_col = np.array([[[-0.5, 0, 0.5]]]).astype('float32')
    kernel_col = torch.as_tensor(np.reshape(kernel_col, (1,1)+kernel_col.shape)).cuda()

    kernel_slice = np.array([[[-0.5]], [[0]], [[0.5]]]).astype('float32')
    kernel_slice = torch.as_tensor(np.reshape(kernel_slice, (1,1)+kernel_slice.shape)).cuda()

    img = np.reshape(img, (1,1)+img.shape).astype('float32')
    img_tensor = torch.as_tensor(img).cuda()
    row_diff_tensor = (torch.zeros_like(img_tensor)).squeeze()
    col_diff_tensor = (torch.zeros_like(img_tensor)).squeeze()
    slice_diff_tensor = (torch.zeros_like(img_tensor)).squeeze()

    row_diff_tensor[:, 1:-1, :] = torch.nn.functional.conv3d(img_tensor, kernel_row, bias=None, stride=1, padding = 0).squeeze()
    col_diff_tensor[:, :, 1:-1] = torch.nn.functional.conv3d(img_tensor, kernel_col, bias=None, stride=1, padding = 0).squeeze()
    slice_diff_tensor[1:-1, :, :] = np.sqrt(reg_z_over_reg) * torch.nn.functional.conv3d(img_tensor, kernel_slice, bias=None, stride=1, padding = 0).squeeze()

    # To match CPU explicit versions
    row_diff_tensor[:, :, [0, -1]] = 0
    row_diff_tensor[[0, -1], :, :] = 0
    col_diff_tensor[:, [0,-1],:] = 0
    col_diff_tensor[[0,-1], :,:] = 0
    slice_diff_tensor[:, [0,-1], :] = 0
    slice_diff_tensor[:, :, [0,-1]] = 0

    grad_norms = (torch.zeros_like(img_tensor)).squeeze()
    grad_norms[:, :, :] = torch.sqrt(torch.square(row_diff_tensor[:, :, :])+torch.square(col_diff_tensor[:, :, :])+torch.square(slice_diff_tensor[:, :, :]))
    tv = grad_norms.sum().cpu().detach().numpy().squeeze()
    grad_norms[grad_norms == 0] = np.inf

    G = torch.zeros_like(img_tensor).squeeze()
    G[1:-1, 1:-1, 2:] += col_diff_tensor[1:-1, 1:-1, 1:-1]/grad_norms[1:-1, 1:-1, 1:-1]
    G[1:-1, 1:-1, :-2] += - col_diff_tensor[1:-1, 1:-1, 1:-1]/grad_norms[1:-1, 1:-1, 1:-1]
    G[1:-1, 2:, 1:-1] += row_diff_tensor[1:-1, 1:-1, 1:-1]/grad_norms[1:-1, 1:-1, 1:-1]
    G[1:-1, :-2, 1:-1] += - row_diff_tensor[1:-1, 1:-1, 1:-1]/grad_norms[1:-1, 1:-1, 1:-1]
    G[2:, 1:-1, 1:-1] += slice_diff_tensor[1:-1, 1:-1, 1:-1]/grad_norms[1:-1, 1:-1, 1:-1]
    G[:-2, 1:-1, 1:-1] += - slice_diff_tensor[1:-1, 1:-1, 1:-1]/grad_norms[1:-1, 1:-1, 1:-1]

    G_cpu = G.cpu().detach().numpy().squeeze()
    torch.cuda.empty_cache()
    del G, grad_norms, row_diff_tensor, col_diff_tensor, slice_diff_tensor, img_tensor, kernel_row, kernel_col, kernel_slice

    return (tv, G_cpu)
