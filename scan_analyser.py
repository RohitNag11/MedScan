import pydicom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import os
import cv2 as cv

# load the DICOM files
path = '/Users/rohit/Documents/Imperial/ME4/FYP/Sample Scans/MJM09_MJM010/MJM09_MJM010_Phantom1607,2003840n/R4G1B43W/TDS102WE'
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
    if hasattr(f, 'SliceLocation') and f.ImageOrientationPatient == [1, 0, 0, 0, 1, 0]:
        slices.append(f)
    else:
        skipcount += 1

print("skipped, no SliceLocation: {}".format(skipcount))


# ensure they are in the correct order
slices = sorted(slices, key=lambda s: s.SliceLocation)
print(f'slices[0].ImagePositionPatient: {slices[0].ImagePositionPatient}')
print(f'slices[-1].ImagePositionPatient: {slices[-1].ImagePositionPatient}')

min_true_z, max_true_z = slices[0].SliceLocation, slices[-1].SliceLocation

print(max_true_z)
print(min_true_z)
print(f'min_true_z: {min_true_z}')
z = np.linspace(0,
                max_true_z - min_true_z,
                len(slices),
                endpoint=True)

# pixel aspects, assuming all slices are the same
ps = slices[0].PixelSpacing
ss = slices[0].SliceThickness
ax_aspect = ps[1]/ps[0]
sag_aspect = ps[1]/ss
cor_aspect = ps[0]/ss

# create 3D array
img_shape = slices[0].pixel_array.shape + (len(slices), )
img3d = np.zeros(img_shape)

# fill 3D array with the images from the files
avg_densities = np.zeros(len(slices))
for k, slice in enumerate(slices):
    cross_section = slice.pixel_array
    img3d[:, :, k] = cross_section
    avg_densities[k] = np.mean(cross_section)

avg_densities_grad = np.gradient(avg_densities)
z_cutoff = (np.argmax(avg_densities_grad) + 10,
            np.argmin(avg_densities_grad) - 10)

filtered_avg_densities_grad2 = np.gradient(avg_densities_grad)
filtered_avg_densities_grad2[:z_cutoff[0]] = 0
filtered_avg_densities_grad2[z_cutoff[1]:] = 0

# Plots
x_cut = 300
x_cut_color = 'orange'
y_cut = img_shape[1]//2
y_cut_color = 'lime'
# z_cut = img_shape[2]//2
z_cut = np.argmax(filtered_avg_densities_grad2)
z_cut = 200
max_grad3 = max(filtered_avg_densities_grad2)
z_cut_color = 'red'

fig = plt.figure(layout="constrained")
subfigs = fig.subfigures(1, 2, wspace=0, width_ratios=[2, 1])
subfigs[0].set_facecolor('0.9')
subfigs[0].suptitle(f'Raw CT Scan Pixel Data')
axs0 = subfigs[0].subplots()

axial_img = img3d[:, :, z_cut]
axs0.imshow(axial_img,
            origin='lower',
            aspect=ax_aspect,
            extent=[0, axial_img.shape[0]*ps[0], 0, axial_img.shape[1]*ps[1]])
axs0.set_title(f'Axial Plane, z={z_cut}', c=z_cut_color)
axs0.set_xlabel('x (mm)')
axs0.set_ylabel('y (mm)')
axs0.axhline(y_cut, c=y_cut_color, alpha=0.5)
axs0.axvline(x_cut, c=x_cut_color, alpha=0.5)

subfigs[1].suptitle('Density variations in z-dir')
subfigs[1].supxlabel('z (mm)')
axs1 = subfigs[1].subplots(5, sharex=True)

cor_img = img3d[x_cut, :, :]
axs1[0].imshow(cor_img,
               origin='lower',
               aspect='auto',
               extent=[0, cor_img.shape[1]*ss, 0, cor_img.shape[0]*ps[1]])
axs1[0].set_title(f'Coronal Plane, x={x_cut}', c=x_cut_color)
axs1[0].set_ylabel('y (mm)')
axs1[0].axvline(z_cut, c=z_cut_color, alpha=0.5)
axs1[0].axhline(y_cut, c=y_cut_color, alpha=0.5)

sag_img = img3d[:, y_cut, :]
axs1[1].imshow(sag_img,
               origin='lower',
               aspect='auto',
               extent=[0, cor_img.shape[1]*ss, 0, cor_img.shape[0]*ps[0]])
axs1[1].set_title(f'Sagittal Plane, y={y_cut}', c=y_cut_color)
axs1[1].axvline(z_cut, c=z_cut_color, alpha=0.5)
axs1[1].axhline(x_cut, c=x_cut_color, alpha=0.5)

axs1[2].set_title('Axial Cross-sectional Pixel Intensity')
axs1[2].set_ylabel("p(x,y)")
axs1[2].set_xlim([min(z), max(z)])
axs1[2].plot(z, avg_densities)
axs1[2].axvline(z_cut, c=z_cut_color, alpha=0.5)

axs1[3].set_title('Axial Cross-sectional Pixel Intensity 1st Derivative')
axs1[3].set_ylabel("p'(x,y)")
axs1[3].set_xlim([min(z), max(z)])
axs1[3].plot(z, avg_densities_grad)
axs1[3].axvline(z_cut, c=z_cut_color, alpha=0.5)


axs1[4].set_title(
    'Filtered Axial Cross-sectional Pixel Intensity 2nd Derivative')
axs1[4].set_ylabel("p''(x,y)")
axs1[4].set_xlim([min(z), max(z)])
axs1[4].plot(z, filtered_avg_densities_grad2)
axs1[4].axvline(z_cut, c=z_cut_color, alpha=0.5)
axs1[4].axvspan(min(z), z_cutoff[0], alpha=0.1, color='black')
axs1[4].axvspan(z_cutoff[1], max(z), alpha=0.1, color='black')


plt.show()
