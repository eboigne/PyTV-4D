import numpy as np
import time

from pytv import *

def run_CPU_tests(N = 100, Nz = 20):
    tv_schemes = ['downwind', 'upwind', 'centered', 'hybrid']
    print('\nRunning CPU tests:')
    for tv_scheme in tv_schemes:
        print('\nCPU test for TV scheme: '+str(tv_scheme))
        time_3D = test_tv_G_D_DT_3D(tv_scheme, cpu_only = True, N = N, Nz = Nz)
        test_2D_to_3D(tv_scheme, cpu_only = True, N = N, Nz = Nz)
        test_operator_tranpose(tv_scheme, cpu_only = True, N = N, Nz = Nz)
    print('\nPassed all CPU tests successfully')

def run_GPU_tests(N = 100, Nz = 20, M = [2, 3, 4]):
    tv_schemes = ['downwind', 'upwind', 'centered', 'hybrid']
    print('\nRunning GPU tests:')
    for tv_scheme in tv_schemes:
        print('\nGPU test for TV scheme: '+str(tv_scheme))
        time_3D = test_tv_G_D_DT_3D(tv_scheme, N = N, Nz = Nz)
        test_2D_to_3D(tv_scheme, N = N, Nz = Nz)
        time_4D = test_tv_D_DT_4D(tv_scheme, N = N, Nz = Nz, M = M)
        test_operator_tranpose(tv_scheme, N = N, Nz = Nz, M = M)
    print('\nPassed all GPU tests successfully')

def test_equal(list, tolerance = 1e-5):
    mean_array = np.mean(list, axis = 0).astype('float32')
    test = True
    for array in list:
        test = test and np.allclose(array, mean_array, rtol=tolerance, atol=tolerance, equal_nan=True)
    return test

def test_operator_tranpose(tv_scheme, N = 100, Nz = 20, M = [2, 3, 4], tolerance = 1e-4, cpu_only = False, n_test = 1):


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

    reg_time = 2.0**(-3)
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

    if Nz < 5:
        Nz = 5
    img = np.random.rand(1,N,N)
    img_3D = np.tile(img, [Nz, 1, 1])

    factor_tv = 1.0
    if tv_scheme == 'downwind' or tv_scheme ==  'centered':
        factor_tv = 1 / (Nz - 2)
    elif tv_scheme == 'upwind' or tv_scheme == 'hybrid':
        factor_tv = 1 / (Nz - 1)

    # Direct CPU implementation
    (tv1, G1) = eval('tv_CPU.tv_'+tv_scheme+'(img)')
    if tv_scheme == 'hybrid':
        (tv1_3D, G1_3D) = eval('tv_CPU.tv_'+tv_scheme+'(img_3D, match_2D_form = True)')
    else:
        (tv1_3D, G1_3D) = eval('tv_CPU.tv_'+tv_scheme+'(img_3D)')
    assert test_equal([tv1, tv1_3D * factor_tv]), 'CPU TV values are not equal'
    assert test_equal([G1, G1_3D[1]]), 'CPU Sub-gradient arrays are not equal'

    if not cpu_only:
        # Direct GPU implementation
        (tv2, G2) = eval('tv_GPU.tv_'+tv_scheme+'(img)')
        if tv_scheme == 'hybrid':
            (tv2_3D, G2_3D) = eval('tv_GPU.tv_'+tv_scheme+'(img_3D, match_2D_form = True)')
        else:
            (tv2_3D, G2_3D) = eval('tv_GPU.tv_'+tv_scheme+'(img_3D)')
        assert test_equal([tv2, tv2_3D * factor_tv]), 'GPU TV values are not equal'
        assert test_equal([G2, G2_3D[1]]), 'GPU Sub-gradient arrays are not equal'

    img = np.reshape(img, [1,1,N,N])
    img_3D = np.reshape(img_3D, [Nz,1,N,N])

    if tv_scheme != 'hybrid': # TODO: Implement match_2D_form for hybrid operators
        # Operator CPU implementation
        D3 = eval('tv_operators_CPU.D_'+tv_scheme+'(img)')
        tv3 = eval('tv_operators_CPU.compute_L21_norm(D3)')
        DT_D3 = eval('tv_operators_CPU.D_T_'+tv_scheme+'(D3)')

        D3_3D = eval('tv_operators_CPU.D_'+tv_scheme+'(img_3D)')
        tv3_3D = eval('tv_operators_CPU.compute_L21_norm(D3_3D)')
        DT_D3_3D = eval('tv_operators_CPU.D_T_'+tv_scheme+'(D3_3D)')

        assert test_equal([tv3, tv3_3D * factor_tv]), 'CPU TV operators values are not equal'
        assert test_equal([D3, D3_3D[1,0:2,:,:,:]]), 'CPU D(x) values are not equal'
        assert test_equal([DT_D3, DT_D3_3D[1:2,:,:,:]]), 'CPU D_T(D(x)) values are not equal'

        if not cpu_only:
            # Operator GPU implementation
            D4 = eval('tv_operators_GPU.D_'+tv_scheme+'(img)')
            tv4 = eval('tv_operators_GPU.compute_L21_norm(D4)')
            DT_D3 = eval('tv_operators_GPU.D_T_'+tv_scheme+'(D4)')

            D4_3D = eval('tv_operators_GPU.D_'+tv_scheme+'(img_3D)')
            tv4_3D = eval('tv_operators_GPU.compute_L21_norm(D4_3D)')
            DT_D4_3D = eval('tv_operators_GPU.D_T_'+tv_scheme+'(D4_3D)')

            assert test_equal([tv3, tv3_3D * factor_tv]), 'CPU TV operators values are not equal'
            assert test_equal([D3, D3_3D[1,0:2,:,:,:]]), 'CPU D(x) values are not equal'
            assert test_equal([DT_D3, DT_D3_3D[1:2,:,:,:]]), 'CPU D_T(D(x)) values are not equal'

    print('\t[PASS] \tScheme '+tv_scheme+' 2D vs 3D: Equal values for TV, G, D, DT from different implementations')

    return True

def test_tv_G_D_DT_3D(tv_scheme, N = 100, Nz = 20, tolerance = 1e-5, cpu_only = False):

    img1 = np.random.rand(Nz, N, N)
    img2 = np.reshape(img1, [Nz, 1, N, N])

    # Direct CPU implementation
    tic = time.time()
    (tv1, G1) = eval('tv_CPU.tv_'+tv_scheme+'(img1)')
    time1 = time.time() - tic

    if not cpu_only:
        # Direct GPU implementation
        tic = time.time()
        (tv2, G2) = eval('tv_GPU.tv_'+tv_scheme+'(img1)')
        time2 = time.time() - tic

    # Operator CPU implementation
    tic = time.time()
    D3 = eval('tv_operators_CPU.D_'+tv_scheme+'(img2)')
    tv3 = eval('tv_operators_CPU.compute_L21_norm(D3)')
    time3 = time.time() - tic
    DT_D3 = eval('tv_operators_CPU.D_T_'+tv_scheme+'(D3)')

    if not cpu_only:
        # Operator GPU implementation
        tic = time.time()
        D4 = eval('tv_operators_GPU.D_'+tv_scheme+'(img2, return_pytorch_tensor=True)')
        tv4 = eval('tv_operators_GPU.compute_L21_norm(D4)')
        time4 = time.time() - tic
        DT_D4 = eval('tv_operators_GPU.D_T_'+tv_scheme+'(D4)')
        D4 = D4.cpu().detach().numpy()

    if cpu_only:
        assert test_equal([tv1, tv3]), 'TV values are not equal'
    else:
        assert test_equal([tv1, tv2, tv3, tv4]), 'TV values are not equal'
        assert test_equal([G1, G2]), 'Sub-gradient arrays are not equal'
        assert test_equal([D3, D4]), 'D(x) arrays are not equal'
        assert test_equal([DT_D3, DT_D4]), 'D_T(D(x)) arrays are not equal'
        del D3, D4, DT_D3, DT_D4

    print('\t[PASS] \tScheme '+tv_scheme+' 3D: Equal values for TV, G, D, DT from different implementations')

    if cpu_only:
        return([time1,time3])
    else:
        return([time1,time2, time3, time4])

def test_tv_D_DT_4D(tv_scheme, N = 100, Nz = 20, M = [2, 3, 4], tolerance = 1e-5):

    time3, time4 = 0, 0
    for this_M in M:
        img = np.random.rand(Nz, this_M, N, N)

        # Operator CPU implementation
        tic = time.time()
        D3 = eval('tv_operators_CPU.D_'+tv_scheme+'(img)')
        tv3 = eval('tv_operators_CPU.compute_L21_norm(D3)')
        time3 += time.time() - tic
        DT_D3 = eval('tv_operators_GPU.D_T_'+tv_scheme+'(D3)')

        # Operator GPU implementation
        tic = time.time()
        D4 = eval('tv_operators_GPU.D_'+tv_scheme+'(img, return_pytorch_tensor=True)')
        tv4 = eval('tv_operators_GPU.compute_L21_norm(D4)')
        time4 += time.time() - tic
        DT_D4 = eval('tv_operators_GPU.D_T_'+tv_scheme+'(D4.cuda())')
        D4 = D4.cpu().detach().numpy()

        assert test_equal([tv3, tv4]), 'TV values are not equal'
        assert test_equal([D3, D4]), 'D(x) arrays are not equal'
        assert test_equal([DT_D3, DT_D4]), 'D_T(D(x)) arrays are not equal'

        del D3, D4, DT_D3, DT_D4

    print('\t[PASS] \tScheme '+tv_scheme+' 4D: Equal values for TV, D, DT from different implementations')

    time3 /= len(M)
    time4 /= len(M)

    return([time3,time4])

def test_transpose(operator, operator_transposed, n_rays = 100, n_test = 5, tolerance = 1e-3, dtype = 'float32', verbose = False, nz = 1, M = 1): # Good test
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
