from medscan import (readers as msr,
                     viewers as msv,
                     segmenters as mss)
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import multivariate_normal


def _1gaussian(x, amp1, cen1, sigma1):
    return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2)))


def _3gaussian(x, amp1, cen1, sigma1, amp2, cen2, sigma2, amp3, cen3, sigma3):
    return (_1gaussian(x, amp1, cen1, sigma1) +
            _1gaussian(x, amp2, cen2, sigma2) +
            _1gaussian(x, amp3, cen3, sigma3))


def fit_3gaussian(x, y, p=[30400, 44.7, 2, 36400, 62.3, 2, 17400, 60.8, 10]):
    popt_2gauss, pcov_2gauss = curve_fit(_3gaussian, x, y, p0=p)
    perr_2gauss = np.sqrt(np.diag(pcov_2gauss))
    print(perr_2gauss)
    params = np.array_split(popt_2gauss, 3)
    gauss_fit = np.zeros(len(x))
    for param in params:
        gauss_fit = np.add(gauss_fit, _1gaussian(x, *param))
    return gauss_fit, params


body_CT = msr.DicomCT(
    '/Users/rohit/Documents/Imperial/ME4/FYP/Sample Scans/MJM09_MJM010/MJM09_MJM010_Phantom1607,2003840n/R4G1B43W/TDS102WE')
lt_bone_mesh = msr.BoneMesh(
    '/Users/rohit/Documents/Imperial/ME4/FYP/Sample Scans/MJM09_MJM010/MJM09_2003840N_Left Tibia.stl', 'Left Tibia')

segmenter = mss.SoftTissueSegmenter(body_CT)
segmenter.add_bone_mesh(lt_bone_mesh)

img3d = body_CT.img3d
z_bounds = body_CT.z_bounds
nk = body_CT.nk

segmented_slices = segmenter.segmented_slices[lt_bone_mesh.name]
img3d = np.array([slice[1] for slice in segmented_slices])
z_bounds = segmented_slices[0][0], segmented_slices[-1][0]
nk = len(segmented_slices)


x = np.arange(body_CT.x_bounds[0],
              body_CT.x_bounds[1] + body_CT.dx, body_CT.dx)
i_slices = np.array([img3d[:, :, i] for i in range(body_CT.ni)])
i_means = [np.mean(slice) for slice in i_slices]
i_gauss_fit, i_gauss_params = fit_3gaussian(x,
                                            i_means,
                                            [30400, 44.7, 2,
                                             36600, 63, 2,
                                             20000, 60.8, 10])

y = np.arange(body_CT.y_bounds[0],
              body_CT.y_bounds[1] + body_CT.dy, body_CT.dy)
j_slices = np.array([img3d[:, j, :] for j in range(body_CT.nj)])
j_means = [np.mean(slice) for slice in j_slices]
j_gauss_fit, j_gauss_params = fit_3gaussian(y,
                                            j_means,
                                            [36900, -212.3, 2,
                                             32500, -196.2, 2,
                                             19900, -198.7, 10])

z = np.arange(z_bounds[0],
              z_bounds[1] + body_CT.dz, body_CT.dz)
k_slices = np.array([img3d[k, :, :] for k in range(nk)])
k_means = [np.mean(slice) for slice in k_slices]

print(len(i_means), body_CT.ni, len(x))
print(len(j_means), body_CT.nj, len(y))
print(len(k_means), body_CT.nk, len(z))

fig, ax = plt.subplots(3, 1)
fig.tight_layout(pad=2.0)
fig.suptitle('Avg Pixel Intensity variation in each Anatomical Planes')

ax[0].set_title('Sagittal')
ax[0].set_xlabel('x')
ax[0].set_ylabel('p̂(x)')
ax[0].plot(x, i_means, label='raw data')
ax[0].plot(x, i_gauss_fit, c='r', ls='--', label='gaussian fit')
ax[0].legend()

ax[1].set_title('Coronal')
ax[1].set_xlabel('y')
ax[1].set_ylabel('p̂(y)')
ax[1].plot(y, j_means, label='raw data')
ax[1].plot(y, j_gauss_fit, c='r', ls='--', label='gaussian fit')
ax[1].legend()

ax[2].set_title('Axial')
ax[2].set_xlabel('z')
ax[2].set_ylabel('p̂(z)')
ax[2].plot(z, k_means, label='raw data')
ax[2].legend()

plt.show()


fig, ax = plt.subplots()
m1 = (-1, -1)
s1 = np.eye(2)
k1 = multivariate_normal(mean=m1, cov=s1)

m2 = (1, 1)
s2 = np.eye(2)
k2 = multivariate_normal(mean=m2, cov=s2)

xx, yy = np.meshgrid(x, y)
xxyy = np.c_[xx.ravel(), yy.ravel()]
zz = np.array([i_gauss_fit[body_CT.get_i_index(
    x)] * j_gauss_fit[body_CT.get_j_index(y)] for x, y in xxyy])

img = zz.reshape((int(body_CT.ni), int(body_CT.nj)))
ax.imshow(img)
ax.set_title('Gaussian Resconstruction at z=100')
plt.show()
