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

def tv_hybrid(img, mask = []):
    '''
    Calculates the total variation and a subgradient of the input image img using the hybrid gradient discretization

    Parameters
    ----------
    img : np.ndarray
        The array of the input image data of dimensions N x N.
    mask : np.ndarray
        A mask used to specify regions of the image ot skip for the TV calculation.

    Returns
    -------
    tv : float
        The value of the total variation.
    G : np.ndarray
        A sub-gradient array of the total variation term of dimensions N x N.
    '''

    if mask != []:
        img[~mask] = 0

    # The intensity differences across rows.
    row_diff = np.zeros_like(img)
    row_diff[:-1, :-1] = img[1:, :-1] - img[:-1, :-1]

    # The intensity differences across columns.
    col_diff = np.zeros_like(img)
    col_diff[:-1, :-1] = img[:-1, 1:] - img[:-1, :-1]

    #  Compute the total variation.
    eps = 0
    grad_norms = np.zeros_like(img)
    grad_norms[:-1, :-1] = np.sqrt(np.square(row_diff[:-1, :-1])
                                   + np.square(col_diff[:-1, :-1]) + np.square(row_diff[:-1, 1:])
                                   +np.square(col_diff[1:, :-1])+eps) / np.sqrt(2)
    tv = np.sum(grad_norms)

    # Find a subgradient.
    G = np.zeros_like(img)
    # When non-differentiable, set to 0.
    grad_norms[grad_norms == 0] = np.inf # not necessary if eps > 0

    G[:-1, :-1] = - (row_diff+col_diff)[:-1, :-1]/grad_norms[:-1, :-1]
    G[:-1, 1:] += (col_diff[:-1, :-1] - row_diff[:-1, 1:])/grad_norms[:-1, :-1]
    G[1:, :-1] += (row_diff[:-1, :-1] - col_diff[1:, :-1])/grad_norms[:-1, :-1]
    G[1:, 1:] += (row_diff[:-1, 1:]+col_diff[1:, :-1])/grad_norms[:-1, :-1]

    return (tv, G)

def tv_downwind(img, mask = []):
    '''
    Calculates the total variation and a subgradient of the input image img using the downwind gradient discretization

    Parameters
    ----------
    img : np.ndarray
        The array of the input image data of dimensions N x N.
    mask : np.ndarray
        A mask used to specify regions of the image ot skip for the TV calculation.

    Returns
    -------
    tv : float
        The value of the total variation.
    G : np.ndarray
        A sub-gradient array of the total variation term of dimensions N x N.
    '''

    if mask != []:
        img[~mask] = 0
    
    # The intensity differences across rows.
    row_diff = np.zeros_like(img)
    row_diff[:-1, :-1] = img[1:, :-1] - img[:-1, :-1]

    # The intensity differences across columns.
    col_diff = np.zeros_like(img)
    col_diff[:-1, :-1] = img[:-1, 1:] - img[:-1, :-1]

    #  Compute the total variation.
    eps = 0
    grad_norms = np.zeros_like(img)
    grad_norms[:-1, :-1] = np.sqrt(np.square(row_diff[:-1, 1:])+np.square(col_diff[1:, :-1])+eps)
    tv = np.sum(grad_norms)

    # Find a subgradient.
    G = np.zeros_like(img)
    # When non-differentiable, set to 0.
    grad_norms[grad_norms == 0] = np.inf # not necessary if eps > 0
    
    G[1:, 1:] =  (row_diff[:-1,1:]+col_diff[1:,:-1])/grad_norms[:-1, :-1]
    G[:-1, 1:] += - row_diff[:-1, 1:]/grad_norms[:-1, :-1]
    G[1:, :-1] += - col_diff[1:, :-1]/grad_norms[:-1, :-1]

    return (tv, G)

def tv_upwind(img, mask = []):
    '''
    Calculates the total variation and a subgradient of the input image img using the upwind gradient discretization

    Parameters
    ----------
    img : np.ndarray
        The array of the input image data of dimensions N x N.
    mask : np.ndarray
        A mask used to specify regions of the image ot skip for the TV calculation.

    Returns
    -------
    tv : float
        The value of the total variation.
    G : np.ndarray
        A sub-gradient array of the total variation term of dimensions N x N.
    '''

    if mask != []:
        img[~mask] = 0
    
    # The intensity differences across rows.
    row_diff = np.zeros_like(img)
    row_diff[:-1, :-1] = img[1:, :-1] - img[:-1, :-1]

    # The intensity differences across columns.
    col_diff = np.zeros_like(img)
    col_diff[:-1, :-1] = img[:-1, 1:] - img[:-1, :-1]

    #  Compute the total variation.
    eps = 0
    grad_norms = np.sqrt(np.square(row_diff)+np.square(col_diff)+eps)
    tv = np.sum(grad_norms)

    # Find a subgradient.
    G = np.zeros_like(img)
    # When non-differentiable, set to 0.
    grad_norms[grad_norms == 0] = np.inf # not necessary if eps > 0

    G[:-1, :-1] =  - (row_diff+col_diff)[:-1, :-1]/grad_norms[:-1, :-1]
    G[1:, :-1] += row_diff[:-1, :-1]/grad_norms[:-1, :-1]
    G[:-1, 1:] += col_diff[:-1, :-1]/grad_norms[:-1, :-1]

    return (tv, G)

def tv_centered(img, mask = []):
    '''
    Calculates the total variation and a subgradient of the input image img using the centered gradient discretization

    Parameters
    ----------
    img : np.ndarray
        The array of the input image data of dimensions N x N.
    mask : np.ndarray
        A mask used to specify regions of the image ot skip for the TV calculation.

    Returns
    -------
    tv : float
        The value of the total variation.
    G : np.ndarray
        A sub-gradient array of the total variation term of dimensions N x N.
    '''

    if mask != []:
        img[~mask] = 0

    # The intensity differences across rows.
    col_diff = 0.5 * ( img[1:-1, 2:] - img[1:-1, :-2] )
    # The intensity differences across columns.
    row_diff = 0.5 * ( img[2:, 1:-1] - img[:-2, 1:-1] )

    #  Compute the total variation.
    eps = 0
    grad_norms = np.sqrt(np.square(row_diff)+np.square(col_diff)+eps)
    tv = np.sum(grad_norms)

    # Find a subgradient.
    G = np.zeros_like(img)
    # When non-differentiable, set to 0.
    grad_norms[grad_norms == 0] = np.inf # not necessary if eps > 0

    G[1:-1, 2:] += col_diff/grad_norms
    G[1:-1, :-2] += - col_diff/grad_norms
    G[2:, 1:-1] += row_diff/grad_norms
    G[:-2, 1:-1] += - row_diff/grad_norms

    return (tv, G)
