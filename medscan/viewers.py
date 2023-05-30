import medscan.readers as msr
import medscan.segmenters as mss
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, RadioButtons, RangeSlider
import numpy as np
import trimesh


class SegmentedRegionSliderPlot:
    def __init__(self,
                 body_CT: msr.DicomCT,
                 bone_meshes: list[msr.BoneMesh],
                 callibrate: bool,
                 colors: list[str]):
        self.body_CT = body_CT
        self.bone_meshes = bone_meshes
        self.callibrate = callibrate
        self.fig, ax = plt.subplots()
        self.z_lims = self.__get_z_lims()
        self.init_z = np.mean(self.z_lims)
        init_z_img = body_CT.get_z_image(
            self.init_z, callibrate=self.callibrate)
        self.img = ax.imshow(init_z_img,
                             extent=np.array([body_CT.x_bounds, body_CT.y_bounds]).flatten(), cmap='magma')
        self.polygons = [bone.get_z_section_polygon(self.init_z,
                                                    bone.name,
                                                    colors[i])
                         for i, bone in enumerate(bone_meshes)]
        [ax.add_patch(poly) for poly in self.polygons]
        self.__configure_plot(ax)
        self.fig.subplots_adjust(left=0.15)
        self.z_slider_ax = self.fig.add_axes([0.1, 0.11, 0.02, 0.76])
        self.z_slider = Slider(
            ax=self.z_slider_ax,
            label='z (mm)',
            valmin=self.z_lims[0],
            valmax=self.z_lims[1],
            valinit=self.init_z,
            orientation="vertical"
        )

    def __get_z_lims(self):
        all_z_lims = np.array([mesh.z_bounds for mesh in self.bone_meshes])
        min_valid_z = max(all_z_lims[:, 0])
        max_valid_z = min(all_z_lims[:, 1])
        return (min_valid_z, max_valid_z)

    def __configure_plot(self, ax):
        ax.set_title('Axial Segmentation of Bones from CT Scan')
        ax.set_xlabel('$x$ (mm)')
        ax.set_ylabel('$y$ (mm)')
        ax.legend(loc='lower right')
        cbar = plt.colorbar(self.img)
        cbar.minorticks_on()
        cbar.set_label('Density (HU)')

    def show(self):
        def update(val):
            z = int(val)
            self.img.set_data(self.body_CT.get_z_image(
                z, callibrate=self.callibrate))
            [poly.set_xy(self.bone_meshes[i].get_z_section_points(z))
             for i, poly in enumerate(self.polygons)]
            self.fig.canvas.draw_idle()
        self.z_slider.on_changed(update)
        plt.show()

    def close(self):
        plt.close(self.fig)


class SegmentedImagesSliderPlot:
    def __init__(self, segmenter: mss.SoftTissueSegmenter):
        self.segmenter = segmenter
        self.fig, ax = plt.subplots(len(segmenter.segmented_slices) + 1)
        self.z_lims = self.__get_z_lims()
        self.init_z = np.mean(self.z_lims)
        init_z_raw_img = segmenter.body_CT.get_z_image(self.init_z)
        self.raw_img = ax.imshow(init_z_raw_img, cmap='magma')
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
        ax.set_xlabel('$x$ (mm)')
        ax.set_ylabel('$y$ (mm)')
        ax.legend(loc='lower right')
        cbar = plt.colorbar(self.img)
        cbar.minorticks_on()
        cbar.set_label('Density (HU)')

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


class Bone3DPlot:
    def __init__(self,
                 bone_meshes: list[msr.BoneMesh],
                 colors: list[str],
                 title: str = '3D Bone Meshes'):
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
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z$')
        ax.set_title(title)
        ax.legend()

    def show(self):
        plt.show()

    def close(self):
        plt.close(self.fig)


class PointCloudPlot:
    def __init__(self, point_cloud, normalised=False, title='Point Cloud', s=2.0, a=1.0, showOnlyGraphics=False):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.x, self.y, self.z, self.p = point_cloud.T
        vmin = 0 if normalised else min(self.p)
        vmax = 1 if normalised else max(self.p)
        self.points = self.ax.scatter(self.x, self.y, self.z, c=self.p,
                                      cmap='magma',
                                      s=s,
                                      alpha=a,
                                      vmin=vmin, vmax=vmax)
        self.showOnlyGraphics = showOnlyGraphics
        self.configure_plot(title, normalised)

    def show(self):
        plt.show()

    def close(self):
        plt.close(self.fig)

    def configure_plot(self, title, normalised):
        self.ax.set_box_aspect(
            [np.ptp(self.x), np.ptp(self.y), np.ptp(self.z)])
        self.ax.set_title(title)
        self.ax.set_xlabel('$x$')
        self.ax.set_ylabel('$y$')
        self.ax.set_zlabel('$z$')
        if self.showOnlyGraphics:
            plt.grid(False)
            plt.axis('off')
        else:
            c_bar_title = 'Normalised Density' if normalised else 'Density (HU)'
            cbar = self.fig.colorbar(self.points)
            cbar.set_label(c_bar_title)
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
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.set_title('Tibias Point Cloud')
        self.ax.set_xlabel('$x$')
        self.ax.set_ylabel('$y$')
        self.ax.set_zlabel('$z$')
        cbar = self.fig.colorbar(self.points)
        cbar.set_label(f'Density (HU)')
        cbar.set_alpha(1)
        cbar.draw_all()


class Density4DPlot:
    def __init__(self,
                 segmenter: mss.SoftTissueSegmenter,
                 bone_mesh: msr.BoneMesh,
                 pixel_thres=0,
                 slice_height=1000,
                 point_size=1.0,
                 lw=0,
                 alpha=0.2,
                 cmap='magma'):
        point_cloud = segmenter.segmented_point_clouds[bone_mesh.name]
        self.x, self.y, self.z, self.p = point_cloud.T
        max_z = bone_mesh.z_bounds[1]
        min_z = max_z - slice_height
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.points = self.ax.scatter(self.x[(self.z > min_z) & (self.p > pixel_thres)],
                                      self.y[(self.z > min_z) & (
                                          self.p > pixel_thres)],
                                      self.z[(self.z > min_z) & (
                                          self.p > pixel_thres)],
                                      c=self.p[(self.z > min_z) & (
                                          self.p > pixel_thres)],
                                      s=point_size,
                                      alpha=alpha,
                                      cmap=cmap)
        self.configure_plot()
        plt.show()

    def configure_plot(self):
        self.ax.set_box_aspect(
            [np.ptp(self.x), np.ptp(self.y), np.ptp(self.z)])
        self.ax.set_title('Left Tibia Point Cloud')
        self.ax.set_xlabel('$x$')
        self.ax.set_ylabel('$y$')
        self.ax.set_zlabel('$z$')
        cbar = self.fig.colorbar(self.points)
        cbar.set_label(f'Density (HU)')
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
                 cmap='magma'):
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
        self.ax.set_box_aspect(
            [np.ptp(self.x), np.ptp(self.y), np.ptp(self.z)])
        ax.set_title('Left Tibia Point Cloud')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z$')
        cbar = self.fig.colorbar(self.points)
        cbar.set_label(f'Density (HU)')
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
    def __init__(self, point_cloud, title='Density Threshold Plot', min_value_init=0.5, max_value_init=1):
        self.x = point_cloud[:, 0]
        self.y = point_cloud[:, 1]
        self.z = point_cloud[:, 2]
        self.density = point_cloud[:, 3]
        self.min_density = min(self.density)
        self.max_density = max(self.density)
        self.min_value_init = min_value_init
        self.max_value_init = max_value_init
        self.title = title
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.slider_ax = plt.axes([0.85, 0.1, 0.01, 0.7])
        self.slider = RangeSlider(
            self.slider_ax, "Threshold", self.min_density, self.max_density, [self.min_value_init, self.max_value_init], orientation='vertical')
        self.mask = (self.density > self.slider.val[0]) & (
            self.density < self.slider.val[1])
        self.scatter = self.ax.scatter(self.x[self.mask],
                                       self.y[self.mask],
                                       self.z[self.mask],
                                       c=self.density[self.mask], cmap='magma', s=2, vmin=0, vmax=1)
        self.cbar_ax = self.fig.add_axes(
            [0.9, 0.1, 0.01, 0.7], sharey=self.slider_ax)
        self.cbar = self.fig.colorbar(self.scatter, self.cbar_ax)
        self.__configure_plot()

    def __configure_plot(self):
        self.ax.set_xlabel('$x$')
        self.ax.set_ylabel('$y$')
        self.ax.set_zlabel('$z$')
        self.ax.set_box_aspect(
            [np.ptp(self.x), np.ptp(self.y), np.ptp(self.z)])
        self.ax.set_xlim(min(self.x), max(self.x))
        self.ax.set_ylim(min(self.y), max(self.y))
        self.ax.set_zlim(min(self.z), max(self.z))
        self.ax.set_title(self.title)
        self.cbar.set_label(f'Normalised Density')
        self.cbar.set_alpha(1)

    def show(self):
        def update(val):
            self.mask = (self.density > self.slider.val[0]) & (
                self.density < self.slider.val[1])
            self.ax.clear()
            self.scatter = self.ax.scatter(self.x[self.mask],
                                           self.y[self.mask],
                                           self.z[self.mask],
                                           c=self.density[self.mask], cmap='magma', s=2, vmin=0, vmax=1)
            self.__configure_plot()
            self.fig.canvas.draw_idle()
        self.slider.on_changed(update)
        plt.show()

    def close(self):
        plt.close(self.fig)


class PredictedClustersPlot:
    def __init__(self, X, y_pred, title='Predicted Clusters Plot'):
        self.X = X
        self.y_pred = y_pred
        self.title = title
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.unique_labels = np.unique(self.y_pred)
        color_map = cm.get_cmap("tab20")
        colors = [color_map(i / len(self.unique_labels))
                  for i in range(len(self.unique_labels))]
        for i, label in enumerate(self.unique_labels):
            color = 'k' if label == -1 else colors[i]
            alpha = 0.2 if label == -1 else 1
            self.ax.scatter(self.X[self.y_pred == label, 0],
                            self.X[self.y_pred == label, 1],
                            self.X[self.y_pred == label, 2],
                            s=2,
                            alpha=alpha,
                            color=color,
                            label=str(label))
        self.configure_plot()

    def configure_plot(self):
        self.ax.set_box_aspect(
            [np.ptp(self.X[:, 0]), np.ptp(self.X[:, 1]), np.ptp(self.X[:, 2])])
        self.ax.set_title(self.title)
        self.ax.set_xlabel('$x$')
        self.ax.set_ylabel('$y$')
        self.ax.set_zlabel('$z$')
        plt.legend()

    def show(self):
        plt.show()

    def close(self):
        plt.close(self.fig)


class PointCloudWithPolygonsPlot:
    def __init__(self, point_cloud, polygon_vertices_array, other_lines=None, title='Point Cloud With Polygons Plot', showOnlyGraphics=False):
        self.point_cloud = point_cloud
        self.polygon_vertices_array = polygon_vertices_array
        self.title = title
        self.point_cloud_plot = PointCloudPlot(
            self.point_cloud, normalised=False, title=self.title, s=1, a=0.5)
        self.ax = self.point_cloud_plot.ax
        self.lines = other_lines if other_lines else []
        self.plot_lines()
        self.plot_polygons()
        if showOnlyGraphics:
            plt.grid(False)
            plt.axis('off')

    def plot_polygons(self):
        for vertices in self.polygon_vertices_array:
            polygon = Poly3DCollection([vertices], alpha=0.2, ec='b')
            self.ax.add_collection(polygon)

    def plot_lines(self):
        for line in self.lines:
            self.ax.plot(line[:, 0], line[:, 1], line[:, 2], lw=3)

    def show(self):
        plt.show()

    def close(self):
        plt.close(self.point_cloud_plot.fig)


class GiftWrapPlot:
    def __init__(self, convex_hull_mesh, points, title='Gift Wrap Plot', c='#FF0000', showOnlyGraphics=False):
        self.convex_hull_mesh = convex_hull_mesh
        self.title = title
        point_cloud_plot = PointCloudPlot(points,
                                          normalised=False,
                                          title=title)
        self.fig = point_cloud_plot.fig
        self.ax = point_cloud_plot.ax
        self.ax.plot_trisurf(self.convex_hull_mesh.vertices[:, 0],
                             self.convex_hull_mesh.vertices[:, 1],
                             self.convex_hull_mesh.vertices[:, 2],
                             triangles=self.convex_hull_mesh.faces,
                             ec=f'{c}50',
                             color=f'{c}',
                             alpha=0.2,
                             lw=0.5,
                             antialiased=True)
        self.showOnlyGraphics = showOnlyGraphics
        self.__configure_plot()

    def __configure_plot(self):
        self.ax.set_box_aspect(
            [np.ptp(self.convex_hull_mesh.vertices[:, 0]),
             np.ptp(self.convex_hull_mesh.vertices[:, 1]),
             np.ptp(self.convex_hull_mesh.vertices[:, 2])])
        self.ax.set_xlabel('$x$')
        self.ax.set_ylabel('$y$')
        self.ax.set_zlabel('$z$')
        self.ax.set_title(self.title)
        if self.showOnlyGraphics:
            plt.grid(False)
            plt.axis('off')

    def plot(self):
        plt.show()

    def close(self):
        plt.close(self.fig)


class RoiVisualiser:
    def __init__(self, bone_mesh_mesh, roi_convex_hull_mesh, title='ROI Visualiser', bone_label='Bone', roi_label='ROI'):
        # meshes = [bone_mesh.mesh for bone_mesh in bone_meshes]
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.title = title
        self.fg_color = 'white'
        self.bg_color = 'black'
        self.bone_vertices = bone_mesh_mesh.vertices
        bone_trisurf = self.ax.plot_trisurf(self.bone_vertices[:, 0],
                                            self.bone_vertices[:, 1],
                                            triangles=bone_mesh_mesh.faces,
                                            Z=self.bone_vertices[:, 2],
                                            ec='#867FEA7B',
                                            lw=0.1,
                                            color=f'#231C833C',
                                            label=bone_label)
        peg_trisurf = self.ax.plot_trisurf(roi_convex_hull_mesh.vertices[:, 0],
                                           roi_convex_hull_mesh.vertices[:, 1],
                                           roi_convex_hull_mesh.vertices[:, 2],
                                           triangles=roi_convex_hull_mesh.faces,
                                           ec='#FF0000',
                                           lw=0.1,
                                           color='#FF0000',
                                           alpha=1,
                                           label=roi_label)
        bone_trisurf._edgecolors2d = bone_trisurf._edgecolor3d
        bone_trisurf._facecolors2d = bone_trisurf._facecolor3d
        peg_trisurf._edgecolors2d = peg_trisurf._edgecolor3d
        peg_trisurf._facecolors2d = peg_trisurf._facecolor3d
        self.__config_plot()

    def __config_plot(self):
        self.fig.set_facecolor(self.bg_color)
        self.ax.set_facecolor(self.bg_color)
        self.ax.grid(False)
        # self.ax.w_xaxis.pane.fill = False
        # self.ax.w_yaxis.pane.fill = False
        # self.ax.w_zaxis.pane.fill = False
        self.ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        self.ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        self.ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        # self.ax.auto_scale_xyz(self.scale, self.scale, self.scale)
        self.ax.set_box_aspect(
            [np.ptp(self.bone_vertices[:, 0]), np.ptp(self.bone_vertices[:, 1]), np.ptp(self.bone_vertices[:, 2])])
        self.ax.set_xlabel('$x$')
        self.ax.set_ylabel('$y$')
        self.ax.set_zlabel('$z$')
        self.ax.set_title(self.title, color=self.fg_color)
        self.ax.legend(loc='lower right')

    def show(self):
        plt.show()

    def close(self):
        plt.close(self.fig)


class TriMeshPlot:
    def __init__(self, tri_mesh, title='Tri Mesh Plot', alpha=0.3):
        self.is_list_input = False if isinstance(
            tri_mesh, trimesh.base.Trimesh) else True
        self.tri_mesh = tri_mesh if not self.is_list_input else None
        self.tri_meshes = tri_mesh if self.is_list_input else None
        self.title = title
        self.alpha = alpha
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.__plot_mesh()
        self.__configure_plot()

    def __plot_mesh(self):
        if not self.is_list_input:
            self.ax.plot_trisurf(self.tri_mesh.vertices[:, 0],
                                 self.tri_mesh.vertices[:, 1],
                                 self.tri_mesh.vertices[:, 2],
                                 triangles=self.tri_mesh.faces,
                                 alpha=self.alpha,
                                 lw=1)
        else:
            for tri_mesh in self.tri_meshes:
                self.ax.plot_trisurf(tri_mesh.vertices[:, 0],
                                     tri_mesh.vertices[:, 1],
                                     tri_mesh.vertices[:, 2],
                                     triangles=tri_mesh.faces,
                                     alpha=self.alpha,
                                     lw=1)

    def __configure_plot(self):
        vertices = self.tri_mesh.vertices if not self.is_list_input else self.tri_meshes[
            0].vertices
        self.ax.set_box_aspect(
            [np.ptp(vertices[:, 0]),
             np.ptp(vertices[:, 1]),
             np.ptp(vertices[:, 2])])
        self.ax.set_xlabel('$x$')
        self.ax.set_ylabel('$y$')
        self.ax.set_zlabel('$z$')
        self.ax.set_title(self.title)

    def show(self):
        plt.show()

    def close(self):
        plt.close(self.fig)
