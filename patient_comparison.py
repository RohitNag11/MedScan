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
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor


class BoneStats:
    def __init__(self, study_name):
        self.study_name = study_name
        self.implant_x_sizes = []
        self.implant_y_sizes = []
        self.implant_z_sizes = []
        self.implant_diag_widths = []
        self.implant_y_origin_depths = []
        self.roi_norm_xy_centers = []
        self.roi_cyl_avg_densities = []
        self.roi_cyl_xyz_sizes = []

    def append(
        self,
        implant_x_size,
        implant_y_size,
        implant_z_size,
        implant_diag_width,
        implant_y_origin_depth,
        roi_norm_xy_center,
        roi_cyl_avg_density,
        roi_cyl_xyz_size,
    ):
        self.implant_x_sizes.append(implant_x_size)
        self.implant_y_sizes.append(implant_y_size)
        self.implant_z_sizes.append(implant_z_size)
        self.implant_diag_widths.append(implant_diag_width)
        self.implant_y_origin_depths.append(implant_y_origin_depth)
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
            "implant_z_size": self.implant_z_sizes,
            "implant_diag_width": self.implant_diag_widths,
            "implant_y_origin_depth": self.implant_y_origin_depths,
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
        density_thresh_percentile -= 0.1
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

    # Get the implant roi origin y distance from the top
    implant_y_origin_depth = bone_points_manipulator.cn_origin_y_depth
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
    # roi_cyl_points_plot.close()
    return (
        implant_x_size,
        implant_y_size,
        implant_z_size,
        implant_diag_width,
        implant_y_origin_depth,
        roi_norm_xy_center,
        roi_cyl_norm_avg_density,
        roi_un_cn_cyl_xyz_size,
    )


def per_bone_analysis(bone_stats, body_CT, bone_mesh, desired_peg_vol_ratio):
    bone_CT = msr.BoneCT(body_CT, bone_mesh, roi_depth=20.0, filter_percent=30)
    # if the bone is a tibia:
    if bone_mesh.is_tibia:
        tibia_stats = tibia_analysis(
            body_CT,
            bone_CT,
            init_density_percentile_thresh=99,
            desired_peg_vol_ratio=desired_peg_vol_ratio,
        )
        bone_stats.append(*tibia_stats)


def main(
    patient_ids: list, output_dir: str, study_name: str, desired_peg_vol_ratio: float
):
    bone_stats = BoneStats(study_name)
    for patient_id in patient_ids:
        print(f"******")
        print(f"Analysing {patient_id}...")
        dicom_path, lt_path, rt_path = get_paths(patient_id)
        body_CT, bone_meshes = get_readers(dicom_path, lt_path, rt_path)
        for bone_mesh in bone_meshes:
            per_bone_analysis(bone_stats, body_CT, bone_mesh, desired_peg_vol_ratio)
    # convert data to csv:
    bone_stats.to_csv(output_dir)
    print(f"**********")
    print(
        f"{study_name} analysis complete!\nResults saved to {output_dir}{study_name}.csv"
    )


def analyse_roi_data_multioutput(res_path):
    df = pd.read_csv(res_path)
    # Define the input and output columns
    input_cols = ["implant_x_size", "implant_y_size"]
    output_cols = [
        "implant_y_origin_depth",
        "roi_norm_x_center",
        "roi_norm_y_center" "roi_cyl_x_size",
        "roi_cyl_y_size",
        "roi_cyl_z_size",
    ]
    # Create the input and output dataframes
    X = df[input_cols]
    y = df[output_cols]
    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Define the base estimator - in this case, a Random Forest
    base_est = RandomForestRegressor(random_state=42)
    # Define the multioutput regressor
    mor = MultiOutputRegressor(base_est)
    # Train the model
    mor.fit(X_train, y_train)
    # Predict output using the model
    y_pred = mor.predict(X_test)


def plot_peg_regions_from_data(res_path, prefix="Batch"):
    df = pd.read_csv(res_path)
    implant_x_sizes = df["implant_x_size"]
    implant_y_sizes = df["implant_y_size"]
    implant_y_origin_depths = df["implant_y_origin_depth"]
    roi_x_centers = df["roi_norm_x_center"]
    roi_y_centers = df["roi_norm_y_center"]
    roi_cyl_x_sizes = df["roi_cyl_x_size"]
    roi_cyl_y_sizes = df["roi_cyl_y_size"]
    for i in range(len(df)):
        no = int(i)
        msv.ImplantVisualiser(
            implant_x_size=implant_x_sizes[i],
            implant_y_size=implant_y_sizes[i],
            implant_y_origin_depth=implant_y_origin_depths[i],
            roi_x_center=roi_x_centers[i],
            roi_y_center=roi_y_centers[i],
            roi_cyl_x_size=roi_cyl_x_sizes[i],
            roi_cyl_y_size=roi_cyl_y_sizes[i],
            title=f"{prefix} {no}: Implant Peg Region",
        ).show()


def analyse_roi_results(
    resultsClassifier: msc.ImplantPegResultsClassifier,
    type: str = "2d",
    deg: int = 1,
    input_cols: list[str] = ["implant_x_size", "implant_y_size"],
    output_cols: list[str] = None,
    outlier_thresh: float = 1.5,
):
    if type == "2d":
        resultsClassifier.analyse_data_2d(
            deg=deg, input_columns=input_cols, output_columns=output_cols
        )
    elif type == "3d":
        resultsClassifier.analyse_data_3d(
            deg=deg,
            input_columns=input_cols,
            output_columns=output_cols,
            outlier_thresh=outlier_thresh,
        )
    elif type == "multi":
        resultsClassifier.analyse_data_multi_param(
            deg=deg,
            input_columns=input_cols,
            output_columns=output_cols,
        )


if __name__ == "__main__":
    output_dir = r"data/results/"
    study_name = "patient_roi_comparisons"
    raw_results_path = rf"{output_dir}{study_name}.csv"
    batch_results_path = rf"{output_dir}{study_name}_batches.csv"

    ## Get results:
    # patient_ids = get_patient_ids((3, 9))
    # desired_peg_vol_ratio = 1 / 50
    # main(patient_ids, output_dir, study_name, desired_peg_vol_ratio)

    ## Analyse the results:
    resultsClassifier = msc.ImplantPegResultsClassifier(raw_results_path)
    analyse_roi_results(
        resultsClassifier,
        type="2d",
        deg=2,
        input_cols=["implant_x_size", "implant_y_size"],
    )
    analyse_roi_results(
        resultsClassifier,
        type="3d",
        deg=2,
        input_cols=["implant_x_size", "implant_y_size"],
        outlier_thresh=1.5,
    )
    analyse_roi_results(
        resultsClassifier,
        type="3d",
        deg=1,
        input_cols=["implant_x_size", "implant_y_size"],
        outlier_thresh=1.5,
    )
    # analyse_roi_results(
    #     resultsClassifier,
    #     type="multi",
    #     deg=1,
    #     input_cols=["implant_x_size", "implant_y_size"],
    #     outlier_thresh=1.5,
    # )
    batch_size = 6
    resultsClassifier.analyse_batch_locations(
        deg=1,
        input_columns=["implant_x_size", "implant_y_size"],
        output_columns=[
            "implant_y_origin_depth",
            "roi_norm_x_center",
            "roi_norm_y_center",
            "roi_cyl_x_size",
            "roi_cyl_y_size",
            "roi_cyl_z_size",
        ],
        batch_size=batch_size,
        output_path=batch_results_path,
    )
    resultsClassifier.visualise_output_for_batches(
        deg=1,
        input_columns=["implant_x_size", "implant_y_size"],
        output_columns=[
            "implant_y_origin_depth",
            "roi_norm_x_center",
            "roi_norm_y_center",
            "roi_cyl_x_size",
            "roi_cyl_y_size",
            "roi_cyl_z_size",
        ],
        batch_size=batch_size,
    )
    plot_peg_regions_from_data(batch_results_path, prefix="Batch")

"""
TODO:
- create a multi output model that takes in implant sizes and ouputs roi sizes and locations
- modify ImplantVisualiser to label the implant and roi sizes and locations
"""
