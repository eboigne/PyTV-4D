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
import time
from pytv import *

def run_CPU_tests(N = 100, Nz = 20, M = [2, 3, 4]):
    '''
    A function that runs CPU tests to check PyTV is working properly.
    '''

    A = np.zeros([1,1,5,5])
    A[0,0,2,2] = 1.0

    tv_schemes = ['downwind', 'upwind', 'central', 'hybrid']
    print('\nRunning CPU tests:')
    for tv_scheme in tv_schemes:
        print('\nCPU test for TV scheme: '+str(tv_scheme))
        test_operator_transpose(tv_scheme, cpu_only = True, N = N, Nz = Nz)
        time_3D = test_tv_G_D_DT_3D(tv_scheme, cpu_only = True, N = N, Nz = Nz)
        test_2D_to_3D(tv_scheme, cpu_only = True, N = N, Nz = Nz)
        time_4D = test_tv_D_DT_4D(tv_scheme, cpu_only = True, N = N, Nz = 1, M = M)
        time_4D = test_tv_D_DT_4D(tv_scheme, cpu_only = True, N = N, Nz = Nz, M = M)
        tv, G = eval('tv_CPU.tv_'+tv_scheme+'(A)')
        print('\nTV(A) = '+str(tv))
        print('A subgradient of TV at A is:\n'+str(G))
    print('\nPassed all CPU tests successfully')

def run_GPU_tests(N = 100, Nz = 20, M = [2, 3, 4]):
    '''
    A function that runs GPU tests to check PyTV is working properly.
    '''

    # tv_schemes = ['downwind', 'upwind', 'central', 'hybrid']
    tv_schemes = ['hybrid']
    print('\nRunning GPU tests:')
    for tv_scheme in tv_schemes:
        print('\nGPU test for TV scheme: '+str(tv_scheme))
        time_3D = test_tv_G_D_DT_3D(tv_scheme, N = N, Nz = Nz)
        test_2D_to_3D(tv_scheme, N = N, Nz = Nz)
        time_4D = test_tv_D_DT_4D(tv_scheme, N = N, Nz = 1, M = M)
        time_4D = test_tv_D_DT_4D(tv_scheme, N = N, Nz = Nz, M = M)
        test_operator_transpose(tv_scheme, N = N, Nz = Nz, M = M)
    print('\nPassed all GPU tests successfully')

def test_equal(list, tolerance = 1e-5):
    '''
    A function that tests whether the elements in list are equal to each within a tolerance

    Parameters
    ----------
    list : list
        A list of elements such as float or np.ndarray
    tolerance : float
        The tolerance within which arguments are called equal

    Returns
    -------
    test : boolean
        Whether the input elements are equal
    '''

    mean_array = np.mean(list, axis = 0).astype('float32')
    test = True
    for array in list:
        test = test and np.allclose(array, mean_array, rtol=tolerance, atol=tolerance, equal_nan=True)
    return test

def test_operator_transpose(tv_scheme, N = 100, Nz = 20, M = [2, 3, 4], tolerance = 1e-4, reg_time = 1.0, cpu_only = False, n_test = 1):
    '''
    A function that tests whether the implemented gradients functions D and D_T for the provided scheme are transposed.

    Parameters
    ----------
    tv_scheme : str
        A string of the tested TV scheme, among 'upwind', 'downwind', 'central' or 'hybrid'
    '''

    # 2D CPU Transpose
    D = lambda x: eval('tv_operators_CPU.D_'+tv_scheme+'(np.reshape(x, (1,1,)+x.shape))')
    D_T = lambda y: eval('tv_operators_CPU.D_T_'+tv_scheme+'(y)')
    assert test_transpose(D, D_T, tolerance = tolerance, n_test = n_test, nz = 1), '2D CPU operators: D and D_T not adjunct'

    # 3D CPU Transpose
    D = lambda x: eval('tv_operators_CPU.D_'+tv_scheme+'(np.transpose(np.reshape(x, (1,)+x.shape), [1, 0, 2, 3]))')
    assert test_transpose(D, D_T, tolerance = tolerance, n_test = n_test, nz = Nz), '3D CPU operators: D and D_T not adjunct'

    # 3D CPU Transpose (no reg z)
    D = lambda x: eval('tv_operators_CPU.D_'+tv_scheme+'(np.transpose(np.reshape(x, (1,)+x.shape), [1, 0, 2, 3]), reg_z_over_reg = 0)')
    D_T = lambda y: eval('tv_operators_CPU.D_T_'+tv_scheme+'(y, reg_z_over_reg = 0)')
    assert test_transpose(D, D_T, tolerance = tolerance, n_test = n_test, nz = Nz), '3D CPU operators (no reg z): D and D_T not adjunct'

    if not cpu_only:
        # 2D GPU Transpose
        D = lambda x: eval('tv_operators_GPU.D_'+tv_scheme+'(np.reshape(x, (1,1,)+x.shape))')
        D_T = lambda y: eval('tv_operators_GPU.D_T_'+tv_scheme+'(y)')
        assert test_transpose(D, D_T, tolerance = tolerance, n_test = n_test, nz = 1), '2D GPU operators: D and D_T not adjunct'

        # 3D GPU Transpose
        D = lambda x: eval('tv_operators_GPU.D_'+tv_scheme+'(np.transpose(np.reshape(x, (1,)+x.shape), [1, 0, 2, 3]))')
        assert test_transpose(D, D_T, tolerance = tolerance, n_test = n_test, nz = Nz), '3D GPU operators: D and D_T not adjunct'

        # 3D GPU Transpose (no reg z)
        D = lambda x: eval('tv_operators_GPU.D_'+tv_scheme+'(np.transpose(np.reshape(x, (1,)+x.shape), [1, 0, 2, 3]), reg_z_over_reg = 0)')
        D_T = lambda y: eval('tv_operators_GPU.D_T_'+tv_scheme+'(y, reg_z_over_reg = 0)')
        assert test_transpose(D, D_T, tolerance = tolerance, n_test = n_test, nz = Nz), '3D GPU operators (no reg z): D and D_T not adjunct'

    for this_M in M:

        # 2D & time CPU Transpose
        D = lambda x: eval('tv_operators_CPU.D_'+tv_scheme+'(np.reshape(x, (1,)+x.shape), reg_time = '+str(reg_time)+')')
        D_T = lambda y: eval('tv_operators_CPU.D_T_'+tv_scheme+'(y, reg_time = '+str(reg_time)+')')
        assert test_transpose(D, D_T, tolerance = tolerance, n_test = n_test, nz = 1, M = this_M), '2D & time CPU operators: D and D_T not adjunct'

        # 3D & time CPU Transpose
        D = lambda x: eval('tv_operators_CPU.D_'+tv_scheme+'(x, reg_time = '+str(reg_time)+')')
        D_T = lambda y: eval('tv_operators_CPU.D_T_'+tv_scheme+'(y, reg_time = '+str(reg_time)+')')
        assert test_transpose(D, D_T, tolerance = tolerance, n_test = n_test, nz = Nz, M = this_M), '3D & time CPU operators: D and D_T not adjunct'

        # 3D & time CPU Transpose (no reg z)
        D = lambda x: eval('tv_operators_CPU.D_'+tv_scheme+'(x, reg_z_over_reg = 0, reg_time = '+str(reg_time)+')')
        D_T = lambda y: eval('tv_operators_CPU.D_T_'+tv_scheme+'(y, reg_z_over_reg = 0, reg_time = '+str(reg_time)+')')
        assert test_transpose(D, D_T, tolerance = tolerance, n_test = n_test, nz = Nz, M = this_M), '3D & time CPU operators (no reg z): D and D_T not adjunct'

        if not cpu_only:
            # 2D & time GPU Transpose
            D = lambda x: eval('tv_operators_GPU.D_'+tv_scheme+'(np.reshape(x, (1,)+x.shape), reg_time = '+str(reg_time)+')')
            D_T = lambda y: eval('tv_operators_GPU.D_T_'+tv_scheme+'(y, reg_time = '+str(reg_time)+')')
            assert test_transpose(D, D_T, tolerance = tolerance, n_test = n_test, nz = 1, M = this_M), '2D & time GPU operators: D and D_T not adjunct'

            # 3D & time GPU Transpose
            D = lambda x: eval('tv_operators_GPU.D_'+tv_scheme+'(x, reg_time = '+str(reg_time)+')')
            D_T = lambda y: eval('tv_operators_GPU.D_T_'+tv_scheme+'(y, reg_time = '+str(reg_time)+')')
            assert test_transpose(D, D_T, tolerance = tolerance, n_test = n_test, nz = Nz, M = this_M), '3D & time GPU operators: D and D_T not adjunct'

            # 3D & time GPU Transpose (no reg z)
            D = lambda x: eval('tv_operators_GPU.D_'+tv_scheme+'(x, reg_z_over_reg = 0, reg_time = '+str(reg_time)+')')
            D_T = lambda y: eval('tv_operators_GPU.D_T_'+tv_scheme+'(y, reg_z_over_reg = 0, reg_time = '+str(reg_time)+')')
            assert test_transpose(D, D_T, tolerance = tolerance, n_test = n_test, nz = Nz, M = this_M), '3D & time GPU operators (no reg z): D and D_T not adjunct'

    print('\t[PASS] \tScheme '+tv_scheme+' operator transposition: D and D_T are adjunct for all tested cases')

    return True

def test_2D_to_3D(tv_scheme, N = 100, Nz = 20, tolerance = 1e-5, cpu_only = False):
    '''
    A function that tests whether the 2D and 3D function implementations provide the same results for the given scheme.

    Parameters
    ----------
    tv_scheme : str
        A string of the tested TV scheme, among 'upwind', 'downwind', 'central' or 'hybrid'
    '''

    if Nz < 5:
        Nz = 5
    img = np.random.rand(1,1,N,N)
    img_3D = np.tile(img, [Nz, 1, 1, 1])
    factor_tv = 1.0 / Nz

    # Direct CPU implementation
    (tv1, G1) = eval('tv_CPU.tv_'+tv_scheme+'(img)')
    if tv_scheme == 'hybrid':
        (tv1_3D, G1_3D) = eval('tv_CPU.tv_'+tv_scheme+'(img_3D, match_2D_form = True)')
    else:
        (tv1_3D, G1_3D) = eval('tv_CPU.tv_'+tv_scheme+'(img_3D)')
    assert test_equal([tv1, tv1_3D * factor_tv], tolerance = tolerance), 'CPU TV values are not equal'
    assert test_equal([G1, np.reshape(G1_3D[1], G1.shape)], tolerance = tolerance), 'CPU Sub-gradient arrays are not equal'

    if not cpu_only:
        # Direct GPU implementation
        (tv2, G2) = eval('tv_GPU.tv_'+tv_scheme+'(img)')
        if tv_scheme == 'hybrid':
            (tv2_3D, G2_3D) = eval('tv_GPU.tv_'+tv_scheme+'(img_3D, match_2D_form = True)')
        else:
            (tv2_3D, G2_3D) = eval('tv_GPU.tv_'+tv_scheme+'(img_3D)')
        assert test_equal([tv2, tv2_3D * factor_tv], tolerance = tolerance), 'GPU TV values are not equal'
        assert test_equal([G2, np.reshape(G2_3D[1], G2.shape)], tolerance = tolerance), 'GPU Sub-gradient arrays are not equal'

    if tv_scheme != 'hybrid': # TODO: Implement match_2D_form for hybrid operators
        # Operator CPU implementation
        D3 = eval('tv_operators_CPU.D_'+tv_scheme+'(img)')
        tv3 = eval('tv_operators_CPU.compute_L21_norm(D3)')
        DT_D3 = eval('tv_operators_CPU.D_T_'+tv_scheme+'(D3)')

        D3_3D = eval('tv_operators_CPU.D_'+tv_scheme+'(img_3D)')
        tv3_3D = eval('tv_operators_CPU.compute_L21_norm(D3_3D)')
        DT_D3_3D = eval('tv_operators_CPU.D_T_'+tv_scheme+'(D3_3D)')

        assert test_equal([tv3, tv3_3D * factor_tv], tolerance = tolerance), 'CPU TV operators values are not equal'
        assert test_equal([D3, D3_3D[1,0:2,:,:,:]], tolerance = tolerance), 'CPU D(x) values are not equal'
        assert test_equal([DT_D3, DT_D3_3D[1:2,:,:,:]], tolerance = tolerance), 'CPU D_T(D(x)) values are not equal'

        if not cpu_only:
            # Operator GPU implementation
            D4 = eval('tv_operators_GPU.D_'+tv_scheme+'(img)')
            tv4 = eval('tv_operators_GPU.compute_L21_norm(D4)')
            DT_D3 = eval('tv_operators_GPU.D_T_'+tv_scheme+'(D4)')

            D4_3D = eval('tv_operators_GPU.D_'+tv_scheme+'(img_3D)')
            tv4_3D = eval('tv_operators_GPU.compute_L21_norm(D4_3D)')
            DT_D4_3D = eval('tv_operators_GPU.D_T_'+tv_scheme+'(D4_3D)')

            assert test_equal([tv3, tv3_3D * factor_tv], tolerance = tolerance), 'CPU TV operators values are not equal'
            assert test_equal([D3, D3_3D[1,0:2,:,:,:]], tolerance = tolerance), 'CPU D(x) values are not equal'
            assert test_equal([DT_D3, DT_D3_3D[1:2,:,:,:]], tolerance = tolerance), 'CPU D_T(D(x)) values are not equal'

    print('\t[PASS] \tScheme '+tv_scheme+' 2D vs 3D: Equal values for TV, G, D, DT from different implementations')

    return True

def test_tv_G_D_DT_3D(tv_scheme, N = 100, Nz = 20, tolerance = 1e-5, cpu_only = False):
    '''
    A function that tests whether the different TV scheme implementations provide the same results for 3D data.

    Parameters
    ----------
    tv_scheme : str
        A string of the tested TV scheme, among 'upwind', 'downwind', 'central' or 'hybrid'
    '''

    img = np.random.rand(Nz, N, N)
    img = np.reshape(img, [Nz, 1, N, N])

    # Direct CPU implementation
    tic = time.time()
    (tv1, G1) = eval('tv_CPU.tv_'+tv_scheme+'(img)')
    time1 = time.time() - tic

    if not cpu_only:
        # Direct GPU implementation
        tic = time.time()
        (tv2, G2) = eval('tv_GPU.tv_'+tv_scheme+'(img)')
        time2 = time.time() - tic

    # Operator CPU implementation
    tic = time.time()
    D3 = eval('tv_operators_CPU.D_'+tv_scheme+'(img)')
    tv3 = eval('tv_operators_CPU.compute_L21_norm(D3)')
    time3 = time.time() - tic
    DT_D3 = eval('tv_operators_CPU.D_T_'+tv_scheme+'(D3)')

    if not cpu_only:
        # Operator GPU implementation
        tic = time.time()
        D4 = eval('tv_operators_GPU.D_'+tv_scheme+'(img, return_pytorch_tensor=True)')
        tv4 = eval('tv_operators_GPU.compute_L21_norm(D4)')
        time4 = time.time() - tic
        DT_D4 = eval('tv_operators_GPU.D_T_'+tv_scheme+'(D4)')
        D4 = D4.cpu().detach().numpy()


    if cpu_only:
        assert test_equal([tv1, tv3], tolerance = tolerance), 'TV values are not equal'
    else:
        assert test_equal([tv1, tv2, tv3, tv4], tolerance = tolerance ), 'TV values are not equal'
        assert test_equal([G1, G2], tolerance = tolerance), 'Sub-gradient arrays are not equal'
        assert test_equal([D3, D4], tolerance = tolerance), 'D(x) arrays are not equal'
        assert test_equal([DT_D3, DT_D4], tolerance = tolerance), 'D_T(D(x)) arrays are not equal'
        del D3, D4, DT_D3, DT_D4, G1, G2

    print('\t[PASS] \tScheme '+tv_scheme+' 3D: Equal values for TV, G, D, DT from different implementations')

    if cpu_only:
        return([time1,time3])
    else:
        return([time1,time2, time3, time4])

def test_tv_D_DT_4D(tv_scheme, N = 100, Nz = 20, M = [2, 3, 4], reg_time = 1.0, tolerance = 1e-5, cpu_only = False):
    '''
    A function that tests whether the different TV scheme implementations provide the same results for 3D & time data.

    Parameters
    ----------
    tv_scheme : str
        A string of the tested TV scheme, among 'upwind', 'downwind', 'central' or 'hybrid'
    '''

    time1, time2, time3, time4 = 0, 0, 0, 0
    for this_M in M:
        img = np.random.rand(Nz, this_M, N, N)

        # Direct CPU implementation
        tic = time.time()
        (tv1, G1) = eval('tv_CPU.tv_'+tv_scheme+'(img, reg_time = '+str(reg_time)+')')
        time1 += time.time() - tic

        if not cpu_only:
            # Direct GPU implementation
            tic = time.time()
            (tv2, G2) = eval('tv_GPU.tv_'+tv_scheme+'(img, reg_time = '+str(reg_time)+')')
            time2 += time.time() - tic

        # Operator CPU implementation
        tic = time.time()
        D3 = eval('tv_operators_CPU.D_'+tv_scheme+'(img, reg_time = '+str(reg_time)+')')
        tv3 = eval('tv_operators_CPU.compute_L21_norm(D3)')
        time3 += time.time() - tic
        DT_D3 = eval('tv_operators_CPU.D_T_'+tv_scheme+'(D3, reg_time = '+str(reg_time)+')')

        if not cpu_only:
            # Operator GPU implementation
            tic = time.time()
            D4 = eval('tv_operators_GPU.D_'+tv_scheme+'(img, return_pytorch_tensor=True, reg_time = '+str(reg_time)+')')
            tv4 = eval('tv_operators_GPU.compute_L21_norm(D4)')
            time4 += time.time() - tic
            DT_D4 = eval('tv_operators_GPU.D_T_'+tv_scheme+'(D4.cuda(), reg_time = '+str(reg_time)+')')
            D4 = D4.cpu().detach().numpy()

        if cpu_only:
            assert test_equal([tv1, tv3], tolerance = tolerance), 'TV values are not equal'
            del D3, DT_D3
        else:
            assert test_equal([tv1, tv2, tv3, tv4], tolerance = tolerance), 'TV values are not equal'
            assert test_equal([G1, G2], tolerance = tolerance), 'Sub-gradient arrays are not equal'
            assert test_equal([D3, D4], tolerance = tolerance), 'D(x) arrays are not equal'
            assert test_equal([DT_D3, DT_D4], tolerance = tolerance), 'D_T(D(x)) arrays are not equal'
            del D3, D4, DT_D3, DT_D4, G1, G2
    print('\t[PASS] \tScheme '+tv_scheme+' 4D: Equal values for TV, D, DT from different implementations')

    time1 /= len(M)
    time2 /= len(M)
    time3 /= len(M)
    time4 /= len(M)

    return([time1, time2, time3,time4])

def test_transpose(operator, operator_transposed, n_rays = 100, n_test = 5, tolerance = 1e-3, dtype = 'float32', verbose = False, nz = 1, M = 1):
    '''
    A function that tests whether the two input operators are numerically tranpose of each other.

    Returns
    -------
    res : boolean
        Whether the input operators are adjunct
    '''

    res = True
    count_wrong = 0

    X = np.squeeze(np.random.randn(nz, M, n_rays, n_rays)) # (Nz) x (M) x N x N
    Y_shape = operator(X)

    for i in range(n_test):

        # Generate arrays
        X = np.random.randn(*X.shape).astype(dtype) # Eg: N x N
        Y = np.random.randn(*Y_shape.shape).astype(dtype) # Eg: 4 X N x N

        # Element-wise multiplications
        dot_product_1 = np.sum(Y * operator(X))
        dot_product_2 = np.sum(X * np.squeeze(operator_transposed(Y)))

        mean_dot_product = 0.5 * (np.abs(dot_product_1) + np.abs(dot_product_2))

        if verbose:
            print(dot_product_1, dot_product_2, "{0:.20f}".format(dot_product_2 - dot_product_1))

        if np.abs((dot_product_1 - dot_product_2)/mean_dot_product) > tolerance:
            count_wrong += 1
            res = False

    if verbose:
        if res:
            print('Transposition test: PASSED')
        else:
            print('Transposition test: FAILED')

    return res

def test_subgradient_descent(tv_scheme, N = 20, Nz = 5, M = 3, tolerance = 1e-5, cpu = True):

    import tifffile, os
    path_save = os.getcwd()

    nb_it = 1000
    step_init = 0.05
    alpha = 0.9
    n_it_reduce = 50
    reg = np.power(2.0,-5)

    tv_table = []
    np.random.seed(0)
    img = np.random.rand(Nz, M, N, N)

    for it in range(nb_it):
        if cpu:
            this_tv, G_tv = eval('tv_CPU.tv_'+tv_scheme+'(img, reg_z_over_reg=1.0, reg_time=1.0)')
        else:
            this_tv, G_tv = eval('tv_GPU.tv_'+tv_scheme+'(img, reg_z_over_reg=1.0, reg_time=1.0)')
        if it < n_it_reduce:
            step = step_init
        else:
            step = step_init * (n_it_reduce/it) ** alpha
        update = - step * reg * G_tv

        if it % 10 == 0:
            for t in range(img.shape[1]):
                tifffile.imsave(path_save+'/save_temp/'+str(t)+'/img_'+str(it).zfill(6)+'.tif', img[Nz//2,t,:,:].astype('float32'))
            print(it, '\t', this_tv, '\t', this_tv / (Nz * M * N * N), '\t', step)

        img += update
        tv_table.append(tv_table)

if __name__ == '__main__':
    run_CPU_tests()
    run_GPU_tests()
