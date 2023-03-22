from medscan import (readers as msr,
                     viewers as msv,
                     segmenters as mss,
                     manipulators as msm,
                     analysers as msa,
                     clasifiers as msc)
from medscan.helpers import geometry as geom
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from tqdm import tqdm
import trimesh


def get_patient_ids(id_range: tuple):
    patient_ids = []
    for i in range(id_range[0], id_range[1] + 1, 2):
        second_id = f'0{i + 1}' if i + 1 < 10 else f'{i + 1}'
        patient_ids.append(f'MJM0{i}_MJM{second_id}')
    return patient_ids


def get_readers(dicom_path, lt_path, rt_path):
    body_CT = msr.DicomCT(dicom_path)
    lt_bone_mesh = msr.BoneMesh(lt_path, 'Left Tibia')
    rt_bone_mesh = msr.BoneMesh(rt_path, 'Right Tibia')
    bone_meshes = [lt_bone_mesh, rt_bone_mesh]
    return body_CT, bone_meshes


def get_paths(patient_id):
    dicom_path = f'/Users/rohit/Documents/Imperial/ME4/FYP/SampleScans/{patient_id}/DICOM'
    stl_path = f'/Users/rohit/Documents/Imperial/ME4/FYP/SampleScans/{patient_id}/STLs'
    # split patient_id into two left_id and right_id
    left_id, right_id = patient_id.split('_')
    lt_path = f'{stl_path}/{left_id}.stl'
    rt_path = f'{stl_path}/{right_id}.stl'
    return dicom_path, lt_path, rt_path


def filter_points_by_desired_peg_roi_vol(implant_roi_cn_points,
                                         init_density_percentile_thresh,
                                         desired_peg_vol_ratio):
    density_thresh_percentile = init_density_percentile_thresh
    points_analyser = msa.PointCloudAnalyser(implant_roi_cn_points)
    peg_hull_volume = peg_hull_rel_volume = 0
    implant_hull_volume = msa.ConvexHullAnalyser(implant_roi_cn_points).volume
    while peg_hull_rel_volume <= desired_peg_vol_ratio:
        print(
            f'Trying density threshold percentile: {density_thresh_percentile}')
        min_density_thresh = points_analyser.get_n_percentile(
            density_thresh_percentile)
        thresholded_point_cloud = implant_roi_cn_points[implant_roi_cn_points[:, 3]
                                                        >= min_density_thresh]
        points_classifier = msc.PointCloudClassifier(thresholded_point_cloud)
        filter_2_points = points_classifier.X_filtered_2
        try:
            peg_cn_hull_analyser = msa.ConvexHullAnalyser(filter_2_points)
            peg_hull_volume = peg_cn_hull_analyser.volume
        except:
            pass
        peg_hull_rel_volume = peg_hull_volume / implant_hull_volume
        density_thresh_percentile -= 1
    hull_mesh = geom.convex_hull_to_trimesh(
        peg_cn_hull_analyser.convex_hull_3d)
    cylinder_mesh = geom.create_cylinder_from_trimesh(hull_mesh)
    return hull_mesh, cylinder_mesh


def tibia_analysis(bone_CT, init_density_percentile_thresh, desired_peg_vol_ratio):
    print(f'Analysing {bone_CT.side} Tibia...')
    # Get the implant roi point cloud of the bone's ct data (used for analysis)
    implant_roi_points = bone_CT.implant_roi_points_4d
    # Create a point cloud manipulator for the implant roi point cloud
    bone_points_manipulator = msm.PointCloudManipulator(
        implant_roi_points, bone_CT.side)
    # Center and normalise the implant roi point cloud
    implant_roi_cn_points = bone_points_manipulator.centred_nomalised_points
    roi_hull_mesh, roi_cylinder_mesh = filter_points_by_desired_peg_roi_vol(implant_roi_cn_points,
                                                                            init_density_percentile_thresh,
                                                                            desired_peg_vol_ratio)
    return roi_hull_mesh, roi_cylinder_mesh


def per_bone_analysis(body_CT, bone_mesh):
    bone_CT = msr.BoneCT(body_CT,
                         bone_mesh,
                         roi_depth=20.0,
                         filter_percent=30)
    bone_roi_hull_meshes = []
    bone_cylinder_meshes = []
    # if the bone is a tibia:
    if bone_mesh.name.split()[-1].lower() == 'tibia':
        roi_hull_mesh, roi_cylinder_mesh = tibia_analysis(bone_CT,
                                                          init_density_percentile_thresh=99,
                                                          desired_peg_vol_ratio=1/15)
        bone_roi_hull_meshes.append(roi_hull_mesh)
        bone_cylinder_meshes.append(roi_cylinder_mesh)
    return bone_roi_hull_meshes, bone_cylinder_meshes


def main(patient_ids: list):
    all_roi_hull_meshes = []
    all_cylinder_meshes = []
    for patient_id in patient_ids:
        print(f'******')
        print(f'Analysing {patient_id}...')
        dicom_path, lt_path, rt_path = get_paths(patient_id)
        body_CT, bone_meshes = get_readers(dicom_path, lt_path, rt_path)
        for bone_mesh in bone_meshes:
            bone_roi_hull_meshes, bone_cylinder_meshes = per_bone_analysis(
                body_CT, bone_mesh)
            all_roi_hull_meshes += bone_roi_hull_meshes
            all_cylinder_meshes += bone_cylinder_meshes
    overlay_hull_mesh_plot = msv.TriMeshPlot(
        all_roi_hull_meshes, title=f'All ROIs Hulls', alpha=0.1)
    overlay_hull_mesh_plot.show()
    overlay_hull_mesh_plot.close()

    overlay_cylinder_mesh_plot = msv.TriMeshPlot(
        all_cylinder_meshes, title=f'All ROIs Cylindrical Hulls', alpha=0.1)
    overlay_cylinder_mesh_plot.show()
    overlay_cylinder_mesh_plot.close()

    # Get common zone between all all_roi_hull_meshes:
    def get_common_zone(meshes: list[trimesh.Trimesh]):
        common_zone = meshes[0]
        for mesh in meshes[1:]:
            common_zone = common_zone.intersection(mesh)
        return common_zone

    common_cylinder_mesh = get_common_zone(all_cylinder_meshes)
    msv.TriMeshPlot(common_cylinder_mesh,
                    title=f'Common ROI Cylindrical Hull', alpha=0.5).show()


if __name__ == '__main__':
    print(trimesh.interfaces.blender.exists)
    patient_ids = get_patient_ids((3, 9))
    main(patient_ids)


'''
TODO:
- get common zone between all meshes 
- add progress bars
'''
