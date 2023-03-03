from medscan import (readers as msr,
                     viewers as msv,
                     segmenters as mss,
                     manipulators as msm,
                     analysers as msa,
                     clasifiers as msc)
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

patient_id = 'MJM07_MJM08'
dicom_path = f'/Users/rohit/Documents/Imperial/ME4/FYP/SampleScans/{patient_id}/DICOM'
stl_path = f'/Users/rohit/Documents/Imperial/ME4/FYP/SampleScans/{patient_id}/STLs'
# split patient_id into two left_id and right_id
left_id, right_id = patient_id.split('_')

body_CT = msr.DicomCT(dicom_path)
lt_mesh = msr.BoneMesh(f'{stl_path}/{left_id}.stl', 'Left Tibia')
rt_mesh = msr.BoneMesh(f'{stl_path}/{right_id}.stl', 'Right Tibia')

# lf_mesh = msr.BoneMesh(f'{stl_path}/{left_id}_Left Femur.stl', 'Left Femur')
# rf_mesh = msr.BoneMesh(f'{stl_path}/{right_id}_Right Femur.stl', 'Right Femur')

# bone_meshes = [lt_mesh, rt_mesh, lf_mesh, rf_mesh]
bone_meshes = [lt_mesh, rt_mesh]
colors = ['#37FF00', '#00E5FF', '#21A700', '#0B85E2']

mesh_plot = msv.Bone3DPlot(bone_meshes, colors)
mesh_plot.show()
mesh_plot.close()

segemented_region_slider_plot = msv.SegmentedRegionSliderPlot(body_CT,
                                                              bone_meshes,
                                                              callibrate=True,
                                                              colors=colors)
segemented_region_slider_plot.show()
segemented_region_slider_plot.close()

# Bone Analysis's:
for bone_mesh in bone_meshes:
    bone_CT = msr.BoneCT(body_CT, bone_mesh,
                         roi_depth=20.0,
                         filter_percent=30)
    # print(bone_CT.side)
    bone_plot = msv.PointCloudPlot(bone_CT.all_points_4d,
                                   normalised=False,
                                   title=f'{bone_mesh.name} All Points',
                                   s=0.2,
                                   a=0.05)
    bone_plot.show()
    # if the bone is a tibia:
    if bone_mesh.name.split()[-1].lower() == 'tibia':
        medial_points = bone_CT.medial_points_4d
        implant_roi_points = bone_CT.implant_roi_points_4d
        bone_points_manipulator = msm.PointCloudManipulator(
            implant_roi_points)
        implant_roi_cn_points = bone_points_manipulator.centred_nomalised_points
        points_analyser = msa.PointCloudAnalyser(implant_roi_cn_points)
        min_density_thresh = points_analyser.get_n_percentile(95)
        thresholded_point_cloud = implant_roi_cn_points[implant_roi_cn_points[:, 3]
                                                        >= min_density_thresh]
        points_classifier = msc.PointCloudClassifier(thresholded_point_cloud)
        filter_1_labels = points_classifier.filter_1_labels
        filter_1_points = points_classifier.X_filtered_1
        filter_2_labels = points_classifier.filter_2_labels
        filter_2_points = points_classifier.X_filtered_2
        convex_hull_2d_vertices_by_z, hull_centres, point_centers = points_classifier.sliced_2d_convex_hull()
        convex_hull_3d = points_classifier.convex_hull_3d()
        hull_analyser = msa.ConvexHullAnalyser(convex_hull_3d)
        print(hull_analyser.volume)

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
                                           title=f'{bone_mesh.name} - Filter 1 (Birch)')
        filter_1_plot.show()
        filter_1_plot.close()

        filter_2_clusters_plot = msv.PredictedClustersPlot(
            filter_1_points, filter_2_labels)
        filter_2_clusters_plot.show()
        filter_2_clusters_plot.close()

        filter_2_plot = msv.PointCloudPlot(filter_2_points,
                                           normalised=True,
                                           title=f'{bone_mesh.name} - Filter 2 (DBSCAN)')
        filter_2_plot.show()
        filter_2_plot.close()

        convex_hull_2d_plot = msv.PointCloudWithPolygonsPlot(filter_2_points, convex_hull_2d_vertices_by_z, other_lines=[hull_centres, point_centers],
                                                             title=f'{bone_mesh.name} - Filter 2 (DBSCAN) with Convex Hull')
        convex_hull_2d_plot.show()
        convex_hull_2d_plot.close()

        convex_hull_3d_plot = msv.GiftWrapPlot(convex_hull_3d, filter_2_points)
        convex_hull_3d_plot.plot()
        convex_hull_3d_plot.close()
    # TO DO:
    # - plot femurs as well. Find gap between tibia and femur bone meshes. If the gap is approx 7mm then the segmented points are the cortical bone. Else there cartiledge will also be included in the segmented points.
    # - Find bottom of saddle and remove points above that if the point cloud only includes bone. Else, go 3mm below the saddle point and remove points above that.
    # - Play around with thresholds. Maybe just selecting the the top 95 percentile only give anomalous points. Maybe need to select the top 85 percentile and then remove the top 5 percentile of that.
    # - Quantify bounding shape of the final filtered point cloud. Try convex hull or layered IOP (algorithm used for get stl slice polygons).
    # - Consider midpoint, bounds, depth and volume of the final filtered point cloud. Also consider the final angle of the peg to find the optimal location for the peg.

    # filter_2_points = points_classifier.X_filtered_2
    # msv.PointCloudPlot(filter_2_points,
    #                    f'{bone_mesh.name} - Filter 2 (Closeness)')
# Left Tibia Analysis:
# ***********


# segmenter = mss.SoftTissueSegmenter(body_CT)
# segmenter.add_bone_mesh(lt_mesh)
# segmenter.add_bone_mesh(rt_mesh)

# # combined_tibia_plot = msv.CombinedTibia4DPlot(segmenter, bone_meshes)

# # point_cloud_plot = msv.Density4DPlot(segmenter,
# #                                      rt_mesh,
# #                                      0,
# #                                      1000,
# #                                      0.02,
# #                                      0,
# #                                      0.5)

# lt_medial_point_cloud = segmenter.medial_point_clouds[rt_mesh.name]
# lt_medial_plot = msv.PointCloudPlot(
#     lt_medial_point_cloud, 'LT Medial Region')

# lt_implant_roi_point_cloud = segmenter.implant_roi_point_clouds[rt_mesh.name]
# lt_implant_roi_plot = msv.PointCloudPlot(
#     lt_implant_roi_point_cloud, 'LT Implant Region')

# lt_manipulator = msm.PointCloudManipulator(lt_implant_roi_point_cloud)
# lt_implant_roi_centred_nomalised_points = lt_manipulator.centred_nomalised_points
# lt_implant_roi_centred_nomalised_plot = msv.PointCloudPlot(
#     lt_implant_roi_centred_nomalised_points, 'LT Implant Region (centred and normalised)')
# lt_implant_roi_centred_nomalised_thres_plot = msv.DensityThresholdPlot(
#     lt_implant_roi_centred_nomalised_points)
# lt_implant_roi_centred_nomalised_thres_plot.plot()


# projected_point_cloud = segmenter.project_3d_to_plane(
#     lt_medial_point_cloud[:, :3], np.array([0, 1, 0]))
# top_most_points = segmenter.get_top_most_projected_points(
#     projected_point_cloud, 25)
# print(segmenter.get_implant_x_plane(top_most_points))
# fig, ax = plt.subplots()
# ax.set_aspect('equal')
# ax.set_title('Locating Implant Region in LT')
# ax.set_xlabel('x')
# ax.set_ylabel('z')
# ax.scatter(projected_point_cloud[:, 0],
#            projected_point_cloud[:, 2], s=0.2, c='b', label='LT medial projected points')
# ax.scatter(top_most_points[:, 0],
#            top_most_points[:, 1], s=1, c='r', label='LT medial top-most points')
# ax.axvline(segmenter.get_implant_x_plane(
#     top_most_points), label='implant line')
# ax.legend()
# plt.show()


# lt_segmented_point_cloud = segmenter.get_down_sampled_point_cloud(
#     lt_mesh,
#     (5, 5, 5))

# down_sampled_plot = msv.Density4DSliderPlot(lt_segmented_point_cloud,
#                                             lt_mesh,
#                                             0,
#                                             300)

# lt_normalized_point_cloud = segmenter.get_normalized_point_cloud(lt_mesh)
# normalized_plot = msv.DensitySliderPlot(lt_normalized_point_cloud)
# normalized_plot.plot()

# with open('lt_pointcloud.pickle', 'wb') as output:
#     pickle.dump(lt_bone_segmented_point_cloud, output)

# print('export complete')
