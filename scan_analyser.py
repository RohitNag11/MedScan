import pydicom
import numpy as np
import matplotlib.pyplot as plt
import os

# load the DICOM files
path = 'C:/Users/Rohit/OneDrive - Imperial College London/Documents/Important Documents/Imperial College/Mech Eng/ME4/FYP/Scans/MJM03_MJM04_Phantom1607,2002722m/R3W1B43U/TTCT0MWE'
directory = os.fsencode(path)
files = []

with os.scandir(path) as it:
    for entry in it:
        if entry.is_file() and entry.name != 'VERSION':
            print(f"loading: {entry.path}")
            files.append(pydicom.dcmread(entry.path))

print("file count: {}".format(len(files)))

# skip files with no SliceLocation (eg scout views)
slices = []
skipcount = 0
for f in files:
    if hasattr(f, 'SliceLocation'):
        slices.append(f)
    else:
        skipcount = skipcount + 1

print("skipped, no SliceLocation: {}".format(skipcount))

# ensure they are in the correct order
slices = sorted(slices, key=lambda s: s.SliceLocation)

# pixel aspects, assuming all slices are the same
ps = slices[0].PixelSpacing
print(f'ps: {ps}')
ss = slices[0].SliceThickness
print(f'ss: {ss}')
ax_aspect = ps[1]/ps[0]
print(f'ax_aspect: {ax_aspect}')
sag_aspect = ps[1]/ss
print(sag_aspect)
cor_aspect = ss/ps[0]
print(cor_aspect)

# create 3D array
img_shape = slices[0].pixel_array.shape + (len(slices), )
img3d = np.zeros(img_shape)


# points = []
# densities = []
# fill 3D array with the images from the files
avg_density_per_layer = np.zeros(len(slices))
for k, slice in enumerate(slices):
    cross_section = slice.pixel_array
    img3d[:, :, k] = cross_section
    avg_density_per_layer[k] = np.mean(cross_section)
    # print(k)
    # for j, row in enumerate(cross_section):
    #     for i, density in enumerate(row):
    #         points.append([i * ps[0], j * ps[1], k * ss])
    #         densities.append(density)

# print(f'points: {points}')
# print(f'densities: {densities}')
densities = np.array(img3d).flatten()
x = np.linspace(0, ps[0]*img_shape[0], img_shape[0], endpoint=False)
y = np.linspace(0, ps[1]*img_shape[1], img_shape[1], endpoint=False)
z = np.linspace(0, ss*img_shape[2], img_shape[2], endpoint=False)

# plot density variation in axial direction:
# fig, ax = plt.subplots()
# fig.suptitle(f'Patient {slices[0].PatientName}')
# ax.set_title('Density Variation in the Axial Direction')
# ax.set_ylabel('pixel value (density)')
# ax.set_xlabel('z (mm)')
# ax.plot(z, avg_density_per_layer)
# plt.show()


# plot 3 orthogonal slices
fig, ax = plt.subplots(2, 2)
fig.tight_layout()
fig.suptitle('Planar Slices Pixel Data')
x_cut = img_shape[0]//2
x_cut_color = 'orange'
y_cut = img_shape[1]//2
y_cut_color = 'lime'
z_cut = img_shape[2]//2
z_cut_color = 'red'

ax[0, 0].imshow(img3d[x_cut, :, :].T, origin='lower')
ax[0, 0].set_aspect(cor_aspect)
ax[0, 0].set_title(f'Coronal Plane, x={x_cut}', c=x_cut_color)
ax[0, 0].set_xlabel('y')
ax[0, 0].set_ylabel('z')
ax[0, 0].axhline(z_cut, c=z_cut_color)
ax[0, 0].axvline(y_cut, c=y_cut_color)

ax[0, 1].imshow(img3d[:, y_cut, :], origin='lower')
ax[0, 1].set_aspect(sag_aspect)
ax[0, 1].set_title(f'Sagittal Plane, y={y_cut}', c=y_cut_color)
ax[0, 1].set_xlabel('z')
ax[0, 1].set_ylabel('x')
ax[0, 1].axhline(x_cut, c=x_cut_color)
ax[0, 1].axvline(z_cut, c=z_cut_color)

ax[1, 0].imshow(img3d[:, :, z_cut], origin='lower')
ax[1, 0].set_aspect(ax_aspect)
ax[1, 0].set_title(f'Axial Plane, z={z_cut}', c=z_cut_color)
ax[1, 0].set_xlabel('y')
ax[1, 0].set_ylabel('x')
ax[1, 0].axhline(y_cut, c=x_cut_color)
ax[1, 0].axvline(x_cut, c=y_cut_color)

plt.show()
