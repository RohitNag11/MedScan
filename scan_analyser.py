import pydicom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import os

# load the DICOM files
path = '/Users/rohit/Documents/Imperial/ME4/FYP/Sample Scans/MJM09_MJM010/MJM09_MJM010_Phantom1607,2003840n/R4G1B43W/TDS102WE'
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
    if hasattr(f, 'SliceLocation') and f.ImageOrientationPatient == [1, 0, 0, 0, 1, 0]:
        slices.append(f)
    else:
        skipcount = skipcount + 1

print("skipped, no SliceLocation: {}".format(skipcount))


# ensure they are in the correct order
slices = sorted(slices, key=lambda s: s.SliceLocation)

print(f'slices[0].SliceLocation: {slices[0].SliceLocation}')
print(f'slices[0].SliceThickness: {slices[0].SliceThickness}')
print(f'slices[0].RotationDirection: {slices[0].RotationDirection}')
print(
    f'slices[0].ImageOrientationPatient: {slices[0].ImageOrientationPatient}')
print(f'slices[0].ImagePositionPatient: {slices[0].ImagePositionPatient}')


print(f'slices[-1].SliceLocation: {slices[-1].SliceLocation}')
print(f'slices[-1].SliceThickness: {slices[-1].SliceThickness}')
print(f'slices[-1].RotationDirection: {slices[-1].RotationDirection}')
print(
    f'slices[-1].ImageOrientationPatient: {slices[-1].ImageOrientationPatient}')
print(f'slices[-1].ImagePositionPatient: {slices[-1].ImagePositionPatient}')


true_z = np.linspace(slices[0].SliceLocation,
                     slices[-1].SliceLocation,
                     len(slices),
                     endpoint=True)

# pixel aspects, assuming all slices are the same
ps = slices[0].PixelSpacing
print(f'ps: {ps}')
ss = slices[0].SliceThickness
print(f'ss: {ss}')
ax_aspect = ps[1]/ps[0]
print(f'ax_aspect: {ax_aspect}')
sag_aspect = ps[1]/ss
print(sag_aspect)
cor_aspect = ps[0]/ss
print(cor_aspect)

# create 3D array
img_shape = slices[0].pixel_array.shape + (len(slices), )
img3d = np.zeros(img_shape)


# points = []
# densities = []
# fill 3D array with the images from the files
avg_densities = np.zeros(len(slices))
for k, slice in enumerate(slices):
    cross_section = slice.pixel_array
    img3d[:, :, k] = cross_section
    avg_densities[k] = np.mean(cross_section)

x = np.linspace(0, ps[0]*img_shape[0], img_shape[0], endpoint=False)
y = np.linspace(0, ps[1]*img_shape[1], img_shape[1], endpoint=False)
z = np.linspace(0, ss*img_shape[2], img_shape[2], endpoint=False)

avg_densities_grad = np.gradient(avg_densities)
z_cutoff = (np.argmax(avg_densities_grad) + 10,
            np.argmin(avg_densities_grad) - 10)

filtered_avg_densities_grad3 = np.gradient(avg_densities_grad)
filtered_avg_densities_grad3[:z_cutoff[0]] = 0
filtered_avg_densities_grad3[z_cutoff[1]:] = 0


# plot 3 orthogonal slices
fig, ax = plt.subplots(2, 3)
fig.tight_layout()
fig.suptitle('Planar Slices Pixel Data')
# # adjust the main plot to make room for the sliders
# fig.subplots_adjust(left=0.25, bottom=0.25)
# x_cut = img_shape[0]//2
x_cut = 300
x_cut_color = 'orange'
y_cut = img_shape[1]//2
y_cut_color = 'lime'
# z_cut = img_shape[2]//2
z_cut = int(np.argmax(filtered_avg_densities_grad3) + slices[0].SliceLocation)
max_grad3 = max(filtered_avg_densities_grad3)
print(max_grad3)
z_cut_color = 'red'

# # Make a horizontal slider to control the z_cut.
# axz = fig.add_axes([0.25, 0.1, 0.65, 0.03])
# z_slider = Slider(
#     ax=axz,
#     label='Axial cut (mm)',
#     valmin=0,
#     valmax=200,
#     valinit=z_cut,
# )
cor_img = img3d[x_cut, :, :]
ax[0, 0].imshow(cor_img, origin='lower', aspect=cor_aspect, extent=[
                min(true_z), max(true_z), cor_img.shape[0], 0])
ax[0, 0].set_title(f'Coronal Plane, x={x_cut}', c=x_cut_color)
ax[0, 0].set_xlabel('z (mm)')
ax[0, 0].set_ylabel('y (mm)')
ax[0, 0].axvline(z_cut, c=z_cut_color, alpha=0.5)
ax[0, 0].axhline(y_cut, c=y_cut_color, alpha=0.5)

sag_img = img3d[:, y_cut, :]
ax[0, 1].imshow(sag_img, origin='lower', aspect=sag_aspect, extent=[
                min(true_z), max(true_z), sag_img.shape[0], 0])
ax[0, 1].set_title(f'Sagittal Plane, y={y_cut}', c=y_cut_color)
ax[0, 1].set_xlabel('z (mm)')
ax[0, 1].set_ylabel('x (mm)')
ax[0, 1].axhline(x_cut, c=x_cut_color, alpha=0.5)
ax[0, 1].axvline(z_cut, c=z_cut_color, alpha=0.5)

ax[0, 2].imshow(img3d[:, :, z_cut], aspect=ax_aspect)
ax[0, 2].set_title(f'Axial Plane, z={z_cut}', c=z_cut_color)
ax[0, 2].set_xlabel('x (mm)')
ax[0, 2].set_ylabel('y (mm)')
ax[0, 2].axhline(y_cut, c=y_cut_color, alpha=0.5)
ax[0, 2].axvline(x_cut, c=x_cut_color, alpha=0.5)


ax[1, 0].set_title('Axial Cross-sectional Pixel Intensity')
ax[1, 0].set_ylabel("p(x,y)")
ax[1, 0].set_xlabel('z (mm)')
ax[1, 0].set_xlim([min(true_z), max(true_z)])
ax[1, 0].plot(true_z, avg_densities)
ax[1, 0].axvline(z_cut, c=z_cut_color, alpha=0.5)

ax[1, 1].set_title('Axial Cross-sectional Pixel Intensity 1st Derivative')
ax[1, 1].set_ylabel("p'(x,y)")
ax[1, 1].set_xlabel('z (mm)')
ax[1, 1].set_xlim([min(true_z), max(true_z)])
ax[1, 1].plot(true_z, avg_densities_grad)
ax[1, 1].axvline(z_cut, c=z_cut_color, alpha=0.5)

ax[1, 2].set_title(
    'Filtered Axial Cross-sectional Pixel Intensity 3rd Derivative')
ax[1, 2].set_ylabel("p'''(x,y)")
ax[1, 2].set_xlabel('z (mm)')
ax[1, 2].set_xlim([min(true_z), max(true_z)])
ax[1, 2].plot(true_z, filtered_avg_densities_grad3)
ax[1, 2].axvline(z_cut, c=z_cut_color, alpha=0.5)
ax[1, 2].axvspan(min(true_z), z_cutoff[0] +
                 slices[0].SliceLocation, alpha=0.1, color='black')
ax[1, 2].axvspan(z_cutoff[1] + slices[0].SliceLocation,
                 max(true_z), alpha=0.1, color='black')

plt.show()
