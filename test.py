import pytv
import numpy as np
import matplotlib.pyplot as plt

noise_level = 0.5
nb_it = 150
regularization = 2e-1
step_size = 5e-3 # If step size is too large, loss function may not decrease at every step

lenna_truth = pytv.utils.lenna() # Open Lenna's grayscale image
lenna_noisy = lenna_truth + noise_level * np.random.rand(*lenna_truth.shape) # Add noise
lenna_estimate = np.copy(lenna_noisy)

loss_fct = np.zeros([nb_it,])
for it in range(nb_it): # A simple sub-gradient descent algorithm for image denoising
    tv, G = pytv.tv.tv_centered(lenna_estimate)
    lenna_estimate += - step_size * ((lenna_estimate - lenna_noisy) + regularization * G)
    loss_fct[it] = 0.5 * np.sum(np.square(lenna_estimate - lenna_noisy)) + regularization * tv

plt.figure(1, figsize=[10, 3], dpi = 150)
plt.subplot(1,3,1, title='Truth (no noise)')
plt.imshow(lenna_truth, cmap = plt.get_cmap('gray'))
plt.axis('off')
plt.subplot(1,3,2, title='Noisy input')
plt.axis('off')
plt.imshow(lenna_noisy, cmap = plt.get_cmap('gray'))
plt.subplot(1,3,3, title='Algorithm output')
plt.imshow(lenna_estimate, cmap = plt.get_cmap('gray'))
plt.axis('off')
plt.tight_layout(pad=0.5)

plt.figure(2, figsize=[3, 2], dpi = 75)
plt.plot(loss_fct)
plt.xlabel('Iteration')
plt.ylabel('Loss function')
plt.tight_layout(pad=0.5)
plt.show()


