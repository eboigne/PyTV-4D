import pytv
import time

print(pytv.__file__)
# pytv.run_CPU_tests()
pytv.run_GPU_tests()
# pytv.test_subgradient_descent('central')

# tic = time.time()
# pytv.test_subgradient_descent('upwind', cpu=True)
# print('CPU took: '+str(time.time()-tic))
#
# tic = time.time()
# pytv.test_subgradient_descent('upwind', cpu=False)
# print('GPU took: '+str(time.time()-tic))
