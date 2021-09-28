import numpy as np


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

        mean_dot_product = 0.5 * (dot_product_1 + dot_product_2)

        if verbose:
            print(dot_product_1, dot_product_2, "{0:.20f}".format(dot_product_2 - dot_product_1))

        if np.abs((dot_product_1 - dot_product_2)/mean_dot_product) > tolerance:
            count_wrong += 1
            res = False
    if res:
        print('Transposition test: PASSED')
    else:
        print('Transposition test: FAILED')

    return res
