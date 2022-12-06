import medscan.readers as msr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D


class SegmentedSliderPlot:
    def __init__(self,
                 body_CT: msr.DicomCT,
                 bone_meshes: list[msr.BoneMesh],
                 labels: list[str],
                 colors: list[str]):
        self.body_CT = body_CT
        self.bone_meshes = bone_meshes
        self.fig, ax = plt.subplots()
        self.z_lims = self.__get_z_lims()
        self.init_z = np.mean(self.z_lims)
        init_z_img = body_CT.get_z_image(self.init_z)
        self.img = ax.imshow(init_z_img,
                             extent=np.array([body_CT.x_bounds, body_CT.y_bounds]).flatten(), cmap='magma')
        self.polygons = [bone.get_z_section_polygon(self.init_z,
                                                    labels[i],
                                                    colors[i])
                         for i, bone in enumerate(bone_meshes)]
        [ax.add_patch(poly) for poly in self.polygons]
        self.__configure_plot(ax)
        z_slider = self.__add_z_slider()
        plt.show()

    def __get_z_lims(self):
        all_z_lims = np.array([mesh.z_bounds for mesh in self.bone_meshes])
        min_valid_z = max(all_z_lims[:, 0])
        max_valid_z = min(all_z_lims[:, 1])
        return (min_valid_z, max_valid_z)

    def __configure_plot(self, ax):
        ax.set_title('Axial Segmentation of Bones from CT Scan')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.legend(loc='lower right')
        cbar = plt.colorbar(self.img)
        cbar.minorticks_on()
        cbar.set_label('Pixel Intensities')

    def __add_z_slider(self):
        self.fig.subplots_adjust(left=0.15)
        axk = self.fig.add_axes([0.1, 0.11, 0.02, 0.76])
        z_slider = Slider(
            ax=axk,
            label='z (mm)',
            valmin=self.z_lims[0],
            valmax=self.z_lims[1],
            valinit=self.init_z,
            orientation="vertical"
        )

        def update(val):
            z = int(val)
            self.img.set_data(self.body_CT.get_z_image(z))
            [poly.set_xy(self.bone_meshes[i].get_z_section_points(z))
             for i, poly in enumerate(self.polygons)]
            self.fig.canvas.draw_idle()
        z_slider.on_changed(update)
        return z_slider

    def close(self):
        plt.close(self.fig)


# class CTOverviewPlot:
#     def __init__(self,
#                  body_CT: msr.DicomCT):
#         self.slices = body_CT.axial_slices
#         min_true_z, max_true_z = body_CT.z_bounds
#         # create 3D array
#         self.img3d = self.__get_img_3d()
#         # fill 3D array with the images from the files
#         self.avg_densities = self.__get_avg_densities()
#         self.avg_densities_grad = np.gradient(self.avg_densities)
#         self.z_cutoffs = self.__get_z_cutoffs()
#         self.filtered_avg_densities_grad2 = self.__get_filtered_avg_densities_grad2()

#         x_cut = 300
#         x_cut_color = 'orange'
#         y_cut = np.mean(body_CT.y_bounds)
#         y_cut_color = 'lime'
#         # z_cut = img_shape[2]//2
#         z_cut = np.argmax(self.filtered_avg_densities_grad2)
#         z_cut_color = 'red'

#         self.fig = plt.figure(layout="constrained")
#         subfigs = self.fig.subfigures(1, 2, wspace=0, width_ratios=[2, 1])
#         subfigs[0].set_facecolor('0.9')
#         subfigs[0].suptitle(f'Raw CT Scan Pixel Data')
#         axs0 = subfigs[0].subplots()
#         axial_img = self.img3d[:, :, z_cut]
#         axs0.imshow(axial_img,
#                     origin='lower',
#                     aspect=body_CT.dx / body_CT.dy,)
#         axs0.set_title(f'Axial Plane, z={z_cut}', c=z_cut_color)
#         axs0.set_xlabel('x (mm)')
#         axs0.set_ylabel('y (mm)')
#         axs0.axhline(y_cut, c=y_cut_color, alpha=0.5)
#         axs0.axvline(x_cut, c=x_cut_color, alpha=0.5)
#         plt.show()

#     def __get_img_3d(self):
#         img_shape = self.slices[0].pixel_array.shape + (len(self.slices), )
#         return np.zeros(img_shape)

#     def __get_avg_densities(self):
#         avg_densities = np.zeros(len(self.slices))
#         for k, slice in enumerate(self.slices):
#             cross_section = slice.pixel_array
#             self.img3d[:, :, k] = cross_section
#             avg_densities[k] = np.mean(cross_section)
#         return avg_densities

#     def __get_z_cutoffs(self):
#         return (np.argmax(self.avg_densities_grad) + 10,
#                 np.argmin(self.avg_densities_grad) - 10)

#     def __get_filtered_avg_densities_grad2(self):
#         filtered_avg_densities_grad2 = np.gradient(self.avg_densities_grad)
#         filtered_avg_densities_grad2[:self.z_cutoffs[0]] = 0
#         filtered_avg_densities_grad2[self.z_cutoffs[1]:] = 0
#         return filtered_avg_densities_grad2


class Bone3DPlot:
    def __init__(self,
                 bone_meshes: list[msr.BoneMesh],
                 labels: list[str],
                 colors: list[str]):
        # meshes = [bone_mesh.mesh for bone_mesh in bone_meshes]
        self.fig = plt.figure()
        ax = self.fig.add_subplot(111, projection='3d')
        trisurfs = [ax.plot_trisurf(bone.mesh.vertices[:, 0],
                                    bone.mesh.vertices[:, 1],
                                    triangles=bone.mesh.faces,
                                    Z=bone.mesh.vertices[:, 2],
                                    ec=colors[i],
                                    lw=0.1,
                                    color=f'{colors[i]}50',
                                    label=labels[i])
                    for i, bone in enumerate(bone_meshes)]
        for trisurf in trisurfs:
            trisurf._edgecolors2d = trisurf._edgecolor3d
            trisurf._facecolors2d = trisurf._facecolor3d
        scale = bone_meshes[0].mesh.vertices.flatten()
        ax.auto_scale_xyz(scale, scale, scale)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.legend()
        plt.show()

    def close(self):
        plt.close(self.fig)
