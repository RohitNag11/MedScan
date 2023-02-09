import medscan.readers as msr
import medscan.segmenters as mss
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, RadioButtons
import numpy as np


class SegmentedRegionSliderPlot:
    def __init__(self,
                 body_CT: msr.DicomCT,
                 bone_meshes: list[msr.BoneMesh],
                 colors: list[str]):
        self.body_CT = body_CT
        self.bone_meshes = bone_meshes
        self.fig, ax = plt.subplots()
        self.z_lims = self.__get_z_lims()
        self.init_z = np.mean(self.z_lims)
        init_z_img = body_CT.get_z_image(self.init_z)
        self.img = ax.imshow(init_z_img,
                             extent=np.array([body_CT.x_bounds, body_CT.y_bounds]).flatten(), cmap='turbo')
        self.polygons = [bone.get_z_section_polygon(self.init_z,
                                                    bone.name,
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


class SegmentedImagesSliderPlot:
    def __init__(self, segmenter: mss.SoftTissueSegmenter):
        self.segmenter = segmenter
        self.fig, ax = plt.subplots(len(segmenter.segmented_slices) + 1)
        self.z_lims = self.__get_z_lims()
        self.init_z = np.mean(self.z_lims)
        init_z_raw_img = segmenter.body_CT.get_z_image(self.init_z)
        self.raw_img = ax.imshow(init_z_raw_img, cmap='turbo')
        self.segmented_imgs = [ax.imshow()]
        [ax.add_patch(poly) for poly in self.polygons]
        self.__configure_plot(ax)
        z_slider = self.__add_z_slider()
        plt.show()

    def __get_z_lims(self):
        all_slices = self.segmenter.values()
        all_z_lims = np.array([[slice[0].keys()[0], slice[-1].keys()[0]]
                               for slice in all_slices])
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
                                    label=bone.name)
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


class PointCloudPlot:
    def __init__(self, point_cloud, title='Point Cloud'):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        x, y, z, p = point_cloud.T
        self.points = self.ax.scatter(x, y, z, c=p,
                                      s=0.1,
                                      alpha=0.2)
        self.configure_plot(title)
        plt.show()

    def configure_plot(self, title):
        self.ax.set_aspect('equal')
        self.ax.set_title(title)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        cbar = self.fig.colorbar(self.points)
        cbar.set_label(f'Pixel Intensities (∝ Density)')
        cbar.set_alpha(1)
        cbar.draw_all()


class CombinedTibia4DPlot:
    def __init__(self,
                 segmenter: mss.SoftTissueSegmenter,
                 bone_meshes: list[msr.BoneMesh]):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        for bone_mesh in bone_meshes:
            point_cloud = segmenter.segmented_point_clouds[bone_mesh.name]
            x, y, z, p = point_cloud.T
            self.points = self.ax.scatter(x, y, z, c=p,
                                          s=0.01,
                                          alpha=0.2)
        self.configure_plot()
        plt.show()

    def configure_plot(self):
        self.ax.set_aspect('equal')
        self.ax.set_title('Tibias Point Cloud')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        cbar = self.fig.colorbar(self.points)
        cbar.set_label(f'Pixel Intensities (∝ Density)')
        cbar.set_alpha(1)
        cbar.draw_all()


class Density4DPlot:
    def __init__(self,
                 segmenter: mss.SoftTissueSegmenter,
                 bone_mesh: msr.BoneMesh,
                 pixel_thres=0,
                 slice_height=1000,
                 point_size=0.02,
                 lw=0,
                 alpha=0.2,
                 cmap='turbo'):
        point_cloud = segmenter.segmented_point_clouds[bone_mesh.name]
        x, y, z, p = point_cloud.T
        max_z = bone_mesh.z_bounds[1]
        min_z = max_z - slice_height
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.points = self.ax.scatter(x[(z > min_z) & (p > pixel_thres)],
                                      y[(z > min_z) & (
                                          p > pixel_thres)],
                                      z[(z > min_z) & (
                                          p > pixel_thres)],
                                      c=p[(z > min_z) & (
                                          p > pixel_thres)],
                                      s=point_size,
                                      alpha=alpha,
                                      cmap=cmap)
        self.configure_plot()
        plt.show()

    def configure_plot(self):
        self.ax.set_aspect('equal')
        self.ax.set_title('Left Tibia Point Cloud')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        cbar = self.fig.colorbar(self.points)
        cbar.set_label(f'Pixel Intensities (∝ Density)')
        cbar.set_alpha(1)
        cbar.draw_all()


class Density4DSliderPlot:
    def __init__(self,
                 point_cloud,
                 bone_mesh: msr.BoneMesh,
                 pixel_thres=0,
                 slice_height=1000,
                 point_size=20,
                 alpha=0.3,
                 cmap='turbo'):
        self.x, self.y, self.z, self.p = point_cloud.T
        self.max_z = bone_mesh.z_bounds[1]
        self.min_z = self.max_z - slice_height
        self.fig = plt.figure()
        ax = self.fig.add_subplot(111, projection='3d')
        self.points = ax.scatter(self.x[(self.z > self.min_z) & (self.p > pixel_thres)],
                                 self.y[(self.z > self.min_z) & (
                                     self.p > pixel_thres)],
                                 self.z[(self.z > self.min_z) & (
                                     self.p > pixel_thres)],
                                 c=self.p[(self.z > self.min_z) & (
                                     self.p > pixel_thres)],
                                 s=point_size,
                                 lw=0,
                                 alpha=alpha,
                                 cmap=cmap)
        self.configure_plot(ax)
        pixel_thres_slider = self.__add_pixel_thres_slider()
        plt.show()

    def configure_plot(self, ax):
        ax.set_aspect('equal')
        ax.set_title('Left Tibia Point Cloud')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        cbar = self.fig.colorbar(self.points)
        cbar.set_label(f'Pixel Intensities (∝ Density)')
        cbar.set_alpha(1)
        cbar.draw_all()

    def __add_height_thres_slider(self, max_height_thres):
        self.fig.subplots_adjust(left=0.15)
        axk = self.fig.add_axes([0.1, 0.11, 0.02, 0.76])
        height_thres_slider = Slider(
            ax=axk,
            label='height threshold (mm)',
            valmin=0,
            valmax=max_height_thres,
            valinit=self.init_height_thres,
            orientation="vertical"
        )

        def update(val):
            min_z = self.max_z - val
            print(min_z)
            x = self.x[(self.z > min_z) & (self.p > self.init_pixel_thres)]
            y = self.y[(self.z > min_z) & (self.p > self.init_pixel_thres)]
            z = self.z[(self.z > min_z) & (self.p > self.init_pixel_thres)]
            p = self.p[(self.z > min_z) & (self.p > self.init_pixel_thres)]
            data = np.array([x, y, z, p]).T
            self.points.set_offsets(data[:, :3])
            self.points.set_array(data[:, 3])
            self.fig.canvas.draw_idle()
        height_thres_slider.on_changed(update)
        return height_thres_slider

    def __add_pixel_thres_slider(self):
        self.fig.subplots_adjust(left=0.15)
        axk = self.fig.add_axes([0.1, 0.11, 0.02, 0.76])
        pixel_thres_slider = Slider(
            ax=axk,
            label='Pixel threshold',
            valmin=0,
            valmax=500,
            valinit=0,
            orientation="vertical",
        )

        def update(val):
            x = self.x[(self.z > self.min_z) & (self.p > val)]
            y = self.y[(self.z > self.min_z) & (self.p > val)]
            z = self.z[(self.z > self.min_z) & (self.p > val)]
            p = self.p[(self.z > self.min_z) & (self.p > val)]
            data = np.array([x, y, z, p]).T
            self.points.set_offsets(data[:, :3])
            self.points.set_array(data[:, 3])
            self.fig.canvas.draw_idle()
        pixel_thres_slider.on_changed(update)
        return pixel_thres_slider


class DensityThresholdPlot:
    def __init__(self, point_cloud, title='Density Threshold Plot'):
        self.x = point_cloud[:, 0]
        self.y = point_cloud[:, 1]
        self.z = point_cloud[:, 2]
        self.density = point_cloud[:, 3]
        self.min_density = min(self.density)
        self.max_density = max(self.density)
        self.min_value_init = self.min_density + self.max_density / 2
        self.title = title
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

    def __configure_plot(self, ax):
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        self.ax.set_aspect('equal')
        self.ax.set_xlim(min(self.x), max(self.x))
        self.ax.set_ylim(min(self.y), max(self.y))
        self.ax.set_title(self.title)

    def plot(self):
        s_min = plt.axes([0.25, 0.1, 0.65, 0.03])
        s_max = plt.axes([0.25, 0.15, 0.65, 0.03])

        self.min_density_slider = Slider(
            s_min, 'Min Density', self.min_density, self.max_density, valinit=self.min_value_init)
        self.max_density_slider = Slider(
            s_max, 'Max Density', self.min_density, self.max_density, valinit=self.max_density)
        mask = (self.density > self.min_density_slider.val) & (
            self.density < self.max_density_slider.val)
        self.ax.scatter(self.x[mask], self.y[mask],
                        self.z[mask], c=self.density[mask], s=0.3)
        self.__configure_plot(self.ax)

        def update(val):
            mask = (self.density > self.min_density_slider.val) & (
                self.density < self.max_density_slider.val)
            self.ax.clear()
            self.ax.scatter(self.x[mask], self.y[mask],
                            self.z[mask], c=self.density[mask], s=0.3)
            self.__configure_plot(self.ax)
            self.fig.canvas.draw_idle()

        self.min_density_slider.on_changed(update)
        self.max_density_slider.on_changed(update)

        reset_axes = plt.axes([0.8, 0.025, 0.1, 0.04])
        button = Button(reset_axes, 'Reset', hovercolor='0.975')

        def reset(event):
            self.min_density_slider.reset()
            self.max_density_slider.reset()

        button.on_clicked(reset)
        plt.show()
