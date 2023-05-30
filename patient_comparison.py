from medscan import (
    readers as msr,
    viewers as msv,
    segmenters as mss,
    manipulators as msm,
    analysers as msa,
    clasifiers as msc,
)
from medscan.helpers import geometry as geom
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from tqdm import tqdm
import trimesh
import pandas as pd


class BoneStats:
    def __init__(self, study_name):
        self.study_name = study_name
        self.implant_x_sizes = []
        self.implant_y_sizes = []
        self.implant_diag_widths = []
        self.roi_norm_xy_centers = []
        self.roi_cyl_avg_densities = []
        self.roi_cyl_xyz_sizes = []

    def append(
        self,
        implant_x_size,
        implant_y_size,
        implant_diag_width,
        roi_norm_xy_center,
        roi_cyl_avg_density,
        roi_cyl_xyz_size,
    ):
        self.implant_x_sizes.append(implant_x_size)
        self.implant_y_sizes.append(implant_y_size)
        self.implant_diag_widths.append(implant_diag_width)
        self.roi_cyl_avg_densities.append(roi_cyl_avg_density)
        self.roi_cyl_xyz_sizes.append(roi_cyl_xyz_size)
        self.roi_norm_xy_centers.append(roi_norm_xy_center)

    def __to_np_arrays(self):
        for key, value in self.__dict__.items():
            if isinstance(value, list):
                self.__dict__[key] = np.array(value)

    def to_csv(self, output_dir):
        self.__to_np_arrays()
        d = {
            "implant_x_size": self.implant_x_sizes,
            "implant_y_size": self.implant_y_sizes,
            "implant_diag_width": self.implant_diag_widths,
            "roi_norm_x_center": self.roi_norm_xy_centers[:, 0],
            "roi_norm_y_center": self.roi_norm_xy_centers[:, 1],
            "roi_cyl_avg_density": self.roi_cyl_avg_densities,
            "roi_cyl_x_size": self.roi_cyl_xyz_sizes[:, 0],
            "roi_cyl_y_size": self.roi_cyl_xyz_sizes[:, 1],
            "roi_cyl_z_size": self.roi_cyl_xyz_sizes[:, 2],
        }
        df = pd.DataFrame(data=d)
        df.to_csv(f"{output_dir}/{self.study_name}.csv", index=False)


def get_patient_ids(id_range: tuple):
    patient_ids = []
    for i in range(id_range[0], id_range[1] + 1, 2):
        second_id = f"0{i + 1}" if i + 1 < 10 else f"{i + 1}"
        patient_ids.append(f"MJM0{i}_MJM{second_id}")
    return patient_ids


def get_readers(dicom_path, lt_path, rt_path):
    body_CT = msr.DicomCT(dicom_path)
    lt_bone_mesh = msr.BoneMesh("Left Tibia", lt_path)
    rt_bone_mesh = msr.BoneMesh("Right Tibia", rt_path)
    bone_meshes = [lt_bone_mesh, rt_bone_mesh]
    return body_CT, bone_meshes


def get_paths(patient_id):
    dicom_path = (
        f"/Users/rohit/Documents/Imperial/ME4/FYP/SampleScans/{patient_id}/DICOM"
    )
    stl_path = f"/Users/rohit/Documents/Imperial/ME4/FYP/SampleScans/{patient_id}/STLs"
    # split patient_id into two left_id and right_id
    left_id, right_id = patient_id.split("_")
    lt_path = f"{stl_path}/{left_id}.stl"
    rt_path = f"{stl_path}/{right_id}.stl"
    return dicom_path, lt_path, rt_path


def filter_points_by_desired_peg_roi_vol(
    implant_roi_cn_points, init_density_percentile_thresh, desired_peg_vol_ratio
):
    density_thresh_percentile = init_density_percentile_thresh
    points_analyser = msa.PointCloudAnalyser(implant_roi_cn_points)
    peg_hull_volume = peg_hull_rel_volume = 0
    implant_hull_volume = msa.ConvexHullAnalyser(implant_roi_cn_points).volume
    while peg_hull_rel_volume <= desired_peg_vol_ratio:
        print(f"Trying density threshold percentile: {density_thresh_percentile}")
        min_density_thresh = points_analyser.get_n_percentile(density_thresh_percentile)
        thresholded_point_cloud = implant_roi_cn_points[
            implant_roi_cn_points[:, 3] >= min_density_thresh
        ]
        points_classifier = msc.PointCloudClassifier(thresholded_point_cloud)
        filter_2_points = points_classifier.X_filtered_2
        try:
            peg_cn_hull_analyser = msa.ConvexHullAnalyser(filter_2_points)
            peg_hull_volume = peg_cn_hull_analyser.volume
        except:
            pass
        peg_hull_rel_volume = peg_hull_volume / implant_hull_volume
        density_thresh_percentile -= 1
    return filter_2_points


def tibia_analysis(
    body_CT, bone_CT, init_density_percentile_thresh, desired_peg_vol_ratio
):
    print(f"Analysing {bone_CT.side} Tibia...")
    # Get the implant roi point cloud of the bone's ct data (used for analysis)
    implant_roi_points = bone_CT.implant_roi_points_4d
    # Get the implant size
    implant_x_size, implant_y_size, implant_z_size = bone_CT.get_implant_size()
    implant_diag_width = (implant_x_size**2 + implant_y_size**2) ** 0.5

    # Create a point cloud manipulator for the implant roi point cloud
    bone_points_manipulator = msm.PointCloudManipulator(
        implant_roi_points, bone_CT.side
    )
    # Center and normalise the implant roi point cloud
    implant_roi_cn_points = bone_points_manipulator.centred_nomalised_points

    filter_2_points = filter_points_by_desired_peg_roi_vol(
        implant_roi_cn_points, init_density_percentile_thresh, desired_peg_vol_ratio
    )
    roi_norm_xy_center = filter_2_points.mean(axis=0)[:2]
    filter_2_points_uncentered = geom.translate_space_3d(
        filter_2_points[:, :3],
        bone_points_manipulator.cn_space_bounds,
        bone_points_manipulator.original_space_bounds,
    )
    roi_un_cn_hull_analyser = msa.ConvexHullAnalyser(filter_2_points_uncentered)
    roi_un_cn_hull = roi_un_cn_hull_analyser.convex_hull_3d
    roi_un_cn_hull_mesh = geom.convex_hull_to_trimesh(roi_un_cn_hull)
    roi_un_cn_cyl_mesh = geom.create_cylinder_from_trimesh(roi_un_cn_hull_mesh)
    roi_un_cn_cyl_vol = roi_un_cn_cyl_mesh.volume
    roi_un_cn_cyl_bone_mesh = msr.BoneMesh("roi_cylinder", mesh=roi_un_cn_cyl_mesh)
    roi_un_cn_cyl_bone_ct = msr.BoneCT(body_CT, roi_un_cn_cyl_bone_mesh)
    roi_un_cn_cyl_avg_density = roi_un_cn_cyl_bone_ct.all_points_4d[:, 3].mean()
    roi_cyl_norm_avg_density = geom.map_to_range(
        roi_un_cn_cyl_avg_density,
        bone_points_manipulator.original_density_bounds,
        bone_points_manipulator.cn_density_bounds,
    )
    roi_un_cn_cyl_xyz_size = roi_un_cn_cyl_bone_mesh.xyz_size

    # NOTE: Verification (remove later)
    roi_cyl_un_cn_point_cloud = roi_un_cn_cyl_bone_ct.all_points_4d
    roi_cyl_un_cn_point_cloud[:, 3] = geom.map_to_range(
        roi_cyl_un_cn_point_cloud[:, 3],
        bone_points_manipulator.original_density_bounds,
        bone_points_manipulator.cn_density_bounds,
    )
    print("average density of roi cylinder: ", np.mean(roi_cyl_un_cn_point_cloud[:, 3]))

    # roi_cyl_points_plot = msv.PointCloudPlot(
    #     roi_cyl_un_cn_point_cloud,
    #     normalised=False,
    #     title=f"{roi_un_cn_cyl_bone_mesh.name}",
    #     s=2,
    #     a=0.5,
    # )
    # roi_cyl_points_plot.show()
    # roi_cyl_points_plot.close()
    return (
        implant_x_size,
        implant_y_size,
        implant_diag_width,
        roi_norm_xy_center,
        roi_cyl_norm_avg_density,
        roi_un_cn_cyl_xyz_size,
    )


def per_bone_analysis(bone_stats, body_CT, bone_mesh):
    bone_CT = msr.BoneCT(body_CT, bone_mesh, roi_depth=20.0, filter_percent=30)
    # if the bone is a tibia:
    if bone_mesh.is_tibia:
        tibia_stats = tibia_analysis(
            body_CT,
            bone_CT,
            init_density_percentile_thresh=99,
            desired_peg_vol_ratio=1 / 15,
        )
        bone_stats.append(*tibia_stats)


def main(patient_ids: list, output_dir: str):
    bone_stats = BoneStats(study_name="patient_roi_comparisons")
    for patient_id in patient_ids:
        print(f"******")
        print(f"Analysing {patient_id}...")
        dicom_path, lt_path, rt_path = get_paths(patient_id)
        body_CT, bone_meshes = get_readers(dicom_path, lt_path, rt_path)
        for bone_mesh in bone_meshes:
            per_bone_analysis(bone_stats, body_CT, bone_mesh)
    # convert data to csv:
    bone_stats.to_csv(output_dir)


def analyse_roi_data(res_path):
    df = pd.read_csv(res_path)
    implant_x_sizes = df["implant_x_size"].to_numpy()
    implant_y_sizes = df["implant_y_size"].to_numpy()
    implant_sizes = df["implant_size"].to_numpy()
    roi_cyl_avg_densities = df["roi_cyl_avg_density"].to_numpy()
    roi_cyl_x_sizes = df["roi_cyl_x_size"].to_numpy()
    roi_cyl_y_sizes = df["roi_cyl_y_size"].to_numpy()
    roi_cyl_z_sizes = df["roi_cyl_z_size"].to_numpy()
    roi_norm_x_centers = df["roi_norm_x_center"].to_numpy()
    roi_norm_y_centers = df["roi_norm_y_center"].to_numpy()

    for size_column in ["implant_x_size", "implant_y_size", "implant_size"]:
        other_columns = [col for col in df.columns if col != size_column]
        ncols = np.ceil(len(other_columns) ** 0.5).astype(int)
        nrows = np.ceil(len(other_columns) / ncols).astype(int)
        fig, ax = plt.subplots(ncols=ncols, nrows=nrows, sharex=True)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle(f"{size_column} vs. other factors", fontsize=10)
        for i, col in enumerate(other_columns):
            ax_i, ax_j = i // ncols, i % ncols
            ax[ax_i, ax_j].scatter(df[size_column].to_numpy(), df[col].to_numpy())
            ax[ax_i, ax_j].set_title(f"{size_column} vs. {col}", fontsize=8)
            ax[ax_i, ax_j].set_xlabel(f"{size_column}", fontsize=6)
            ax[ax_i, ax_j].set_ylabel(f"{col}", fontsize=6)
        plt.show()

    # # plot all data against implant size as scatter plots:
    # ncols, nrows = 3, 2
    # fig, ax = plt.subplots(ncols=ncols, nrows=nrows)
    # for i, col in enumerate(df.columns[2:]):
    #     ax_i, ax_j = i // ncols, i % ncols
    #     ax[ax_i, ax_j].scatter(implant_sizes, df[col].to_numpy())
    #     ax[ax_i, ax_j].set_title(f'Implant size vs. {col}')
    #     ax[ax_i, ax_j].set_xlabel('Implant diagonal size (mm)')
    #     ax[ax_i, ax_j].set_ylabel(f'{col}')
    # plt.show()


if __name__ == "__main__":
    output_dir = r"data/results/"
    patient_ids = get_patient_ids((3, 9))
    main(patient_ids, output_dir)
    # results_path = r'data/results/roi_data.csv'
    # analyse_roi_data(results_path)


"""
TODO:
- get common zone between all meshes 
- add progress bars
- work out (averaged?) normalised and true density in roi volume

-Validate true density:
- calibrate (make density space) to 5, 95 percentile of the full knee ct points.
1. get callibrated density from ct scan
2. calibrate normalised densities from literature
- compare the two

- Compare differences in relative peg location for different groups of knee sizes.
    - group data by knee sizes - find average peg location for each group
    - 
    
- Quantify normalised data better. Define the boundaries.
    
- Use mimics to manually segment the additional scans.
"""
