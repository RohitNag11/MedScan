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


def tibia_analysis(bone_mesh, bone_CT, init_density_thresh_percentile, rel_peg_vol_thresh):
    print(f'Analysing {bone_CT.side} Tibia...')
    # Get the medial point cloud of the bone's ct data (used for plotting)
    medial_points = bone_CT.medial_points_4d
    # Get the implant roi point cloud of the bone's ct data (used for analysis)
    implant_roi_points = bone_CT.implant_roi_points_4d
    # Create a point cloud manipulator for the implant roi point cloud
    bone_points_manipulator = msm.PointCloudManipulator(
        implant_roi_points, bone_CT.side)
    # Center and normalise the implant roi point cloud
    implant_roi_cn_points = bone_points_manipulator.centred_nomalised_points
    points_analyser = msa.PointCloudAnalyser(implant_roi_cn_points)
    peg_hull_volume = peg_hull_rel_volume = 0
    density_thresh_percentile = init_density_thresh_percentile
    implant_hull_volume = msa.ConvexHullAnalyser(implant_roi_cn_points).volume
    while peg_hull_rel_volume <= rel_peg_vol_thresh:
        print(
            f'Trying density threshold percentile: {density_thresh_percentile}')
        min_density_thresh = points_analyser.get_n_percentile(
            density_thresh_percentile)
        thresholded_point_cloud = implant_roi_cn_points[implant_roi_cn_points[:, 3]
                                                        >= min_density_thresh]
        points_classifier = msc.PointCloudClassifier(thresholded_point_cloud)
        filter_1_labels = points_classifier.filter_1_labels
        filter_1_points = points_classifier.X_filtered_1
        filter_2_labels = points_classifier.filter_2_labels
        filter_2_points = points_classifier.X_filtered_2
        try:
            peg_cn_hull_analyser = msa.ConvexHullAnalyser(filter_2_points)
            peg_hull_volume = peg_cn_hull_analyser.volume
        except:
            pass
        peg_hull_rel_volume = peg_hull_volume / implant_hull_volume
        density_thresh_percentile -= 1
    print(f'Density Threshold Percentile: {density_thresh_percentile}')
    convex_hull_2d_vertices_by_z, hull_centres, point_centers = peg_cn_hull_analyser.sliced_convex_hull_2d()
    peg_cn_hull_mesh = geom.convex_hull_to_trimesh(
        peg_cn_hull_analyser.convex_hull_3d)
    filter_2_points_uncentered = geom.translate_space_3d(filter_2_points[:, :3],
                                                         bone_points_manipulator.cn_space_bounds,
                                                         bone_points_manipulator.original_space_bounds)
    peg_un_cn_hull_analyser = msa.ConvexHullAnalyser(
        filter_2_points_uncentered)
    peg_un_cn_hull = peg_un_cn_hull_analyser.convex_hull_3d
    peg_un_cn_hull_volume = peg_un_cn_hull_analyser.volume
    peg_un_cn_hull_mesh = geom.convex_hull_to_trimesh(peg_un_cn_hull)
    peg_un_cn_cylinder_mesh = geom.create_cylinder_from_trimesh(
        peg_un_cn_hull_mesh)
    print(f'Filter 2 Convex Hull Volume: {peg_un_cn_hull_volume} mm^3')

    # Plots:
    medial_points_plot = msv.PointCloudPlot(medial_points,
                                            normalised=False,
                                            title=f'{bone_mesh.name} Medial Points',
                                            s=2,
                                            a=0.5)
    medial_points_plot.show()
    medial_points_plot.close()

    implant_roi_points_plot = msv.PointCloudPlot(implant_roi_points,
                                                 normalised=False,
                                                 title=f'{bone_mesh.name} Implant ROI Points',
                                                 s=2,
                                                 a=0.5)
    implant_roi_points_plot.show()
    implant_roi_points_plot.close()

    centred_nomalised_points_plot = msv.PointCloudPlot(implant_roi_cn_points,
                                                       normalised=True,
                                                       title=f'{bone_mesh.name} Implant ROI (centred and normalised)')
    centred_nomalised_points_plot.show()
    centred_nomalised_points_plot.close()

    centred_nomalised_thres_plot = msv.DensityThresholdPlot(
        implant_roi_cn_points,
        f'{bone_mesh.name} Implant ROI (centred and normalised) Thresholded',
        min_density_thresh)
    centred_nomalised_thres_plot.show()
    centred_nomalised_thres_plot.close()

    filter_1_clusters_plot = msv.PredictedClustersPlot(
        thresholded_point_cloud, filter_1_labels)
    filter_1_clusters_plot.show()
    filter_1_clusters_plot.close()

    filter_1_plot = msv.PointCloudPlot(filter_1_points,
                                       normalised=True,
                                       title=f'{bone_mesh.name} Peg ROI - Filter 1 (Birch)')
    filter_1_plot.show()
    filter_1_plot.close()

    filter_2_clusters_plot = msv.PredictedClustersPlot(
        filter_1_points, filter_2_labels)
    filter_2_clusters_plot.show()
    filter_2_clusters_plot.close()

    filter_2_plot = msv.PointCloudPlot(filter_2_points,
                                       normalised=True,
                                       title=f'{bone_mesh.name} Peg ROI - Filter 2 (DBSCAN)')
    filter_2_plot.show()
    filter_2_plot.close()

    convex_hull_2d_plot = msv.PointCloudWithPolygonsPlot(filter_2_points,
                                                         convex_hull_2d_vertices_by_z,
                                                         other_lines=[
                                                             hull_centres, point_centers],
                                                         title=f'{bone_mesh.name} Peg ROI - 2D Sliced Convex Hull')
    convex_hull_2d_plot.show()
    convex_hull_2d_plot.close()

    convex_hull_3d_plot = msv.GiftWrapPlot(
        peg_cn_hull_mesh, filter_2_points)
    convex_hull_3d_plot.plot()
    convex_hull_3d_plot.close()

    overlay_mesh_plot = msv.TriMeshPlot(
        [peg_un_cn_cylinder_mesh, peg_un_cn_hull_mesh], title=f'{bone_mesh.name} Peg ROI - Comparison of Convex Hull and Cylinder Fit')
    overlay_mesh_plot.show()
    overlay_mesh_plot.close()

    roi_visualiser = msv.RoiVisualiser(bone_mesh.mesh,
                                       peg_un_cn_hull_mesh,
                                       title='Approximate Implant Peg ROI Uncentered',
                                       bone_label=bone_mesh.name,
                                       roi_label='Pegs ROI')
    roi_visualiser.show()
    roi_visualiser.close()

    roi_visualiser_cylinder = msv.RoiVisualiser(bone_mesh.mesh,
                                                peg_un_cn_cylinder_mesh,
                                                title='Approximate Implant Peg ROI Cylinder Fit Uncentered',
                                                bone_label=bone_mesh.name,
                                                roi_label='Pegs ROI Cylinder Fit')
    roi_visualiser_cylinder.show()
    roi_visualiser_cylinder.close()


def per_bone_analysis(body_CT, bone_mesh):
    bone_CT = msr.BoneCT(body_CT,
                         bone_mesh,
                         roi_depth=20.0,
                         filter_percent=30)
    bone_plot = msv.PointCloudPlot(bone_CT.all_points_4d,
                                   normalised=False,
                                   title=f'{bone_mesh.name} All Points',
                                   s=0.2,
                                   a=0.05)
    bone_plot.show()
    # if the bone is a tibia:
    if bone_mesh.name.split()[-1].lower() == 'tibia':
        tibia_analysis(bone_mesh,
                       bone_CT,
                       init_density_thresh_percentile=99,
                       rel_peg_vol_thresh=1/15)


def pre_analysis_plots(body_CT, bone_meshes):
    bone_colors = ['#37FF00', '#00E5FF', '#21A700', '#0B85E2']
    # Plot the bone STLS:
    mesh_plot = msv.Bone3DPlot(bone_meshes,
                               colors=bone_colors,
                               title='STL Bone Meshes')
    mesh_plot.show()
    mesh_plot.close()

    # Plot the CT slices with the segmented bone regions:
    segemented_region_slider_plot = msv.SegmentedRegionSliderPlot(body_CT,
                                                                  bone_meshes,
                                                                  callibrate=True,
                                                                  colors=bone_colors)
    segemented_region_slider_plot.show()
    segemented_region_slider_plot.close()


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


def main(patient_id):
    dicom_path, lt_path, rt_path = get_paths(patient_id)
    body_CT, bone_meshes = get_readers(dicom_path, lt_path, rt_path)
    pre_analysis_plots(body_CT, bone_meshes)
    for bone_mesh in bone_meshes:
        per_bone_analysis(body_CT, bone_mesh)


if __name__ == '__main__':
    patient_id = 'MJM03_MJM04'
    main(patient_id)

    # TO do:
    # find bounding box
    # Find average density of bounding box
    # angles -30-60 degrees in y-z plane.
