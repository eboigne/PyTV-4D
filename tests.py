import numpy as np


def test_transpose(operator, operator_transposed, projector_type = '', n_rays = 100, n_test = 5, tolerance = 1e-8, dtype = 'float32', verbose = False): # Good test
    res = True
    count_wrong = 0

    X = np.random.randn(n_rays, n_rays) # N x N

    if projector_type == '':
        Y_shape = operator(X)
    else:
        Y_shape = operator(X, projector_type  = projector_type)

    for i in range(n_test):

        # Generate arrays
        X = np.random.randn(n_rays, n_rays) # Eg: N x N
        Y = np.random.randn(*Y_shape.shape) # Eg: 4 X N x N

        X = X.astype(dtype)
        Y = Y.astype(dtype)

        if projector_type == '':
            dot_product_1 = np.sum(Y * operator(X)) # Element-wise multiplication
            dot_product_2 = np.sum(X * operator_transposed(Y)) # Element-wise multiplication
        else:
            dot_product_1 = np.sum(Y * operator(X, projector_type = projector_type)) # Element-wise multiplication
            dot_product_2 = np.sum(X * operator_transposed(Y, projector_type = projector_type)) # Element-wise multiplication
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