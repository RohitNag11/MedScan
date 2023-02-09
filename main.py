from medscan import (readers as msr,
                     viewers as msv,
                     segmenters as mss,
                     manipulators as msm)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

body_CT = msr.DicomCT(
    '/Users/rohit/Documents/Imperial/ME4/FYP/Sample Scans/MJM09_MJM010/MJM09_MJM010_Phantom1607,2003840n/R4G1B43W/TDS102WE')
lt_mesh = msr.BoneMesh(
    '/Users/rohit/Documents/Imperial/ME4/FYP/Sample Scans/MJM09_MJM010/MJM09_2003840N_Left Tibia.stl', 'Left Tibia')
rt_mesh = msr.BoneMesh(
    '/Users/rohit/Documents/Imperial/ME4/FYP/Sample Scans/MJM09_MJM010/MJM10_2003840N_Right Tibia.stl', 'Right Tibia')

bone_meshes = [lt_mesh, rt_mesh]
colors = ['#000000', '#FFFFFF']

mesh_plot = msv.Bone3DPlot(bone_meshes, colors)
mesh_plot.close()

segemented_region_slider_plot = msv.SegmentedRegionSliderPlot(body_CT,
                                                              bone_meshes,
                                                              colors)
segemented_region_slider_plot.close()


lt_CT = msr.BoneCT(body_CT, lt_mesh, 30.0)
print(lt_CT.side)
msv.PointCloudPlot(lt_CT.all_points_4d, 'LT All Points')
msv.PointCloudPlot(lt_CT.medial_points_4d, 'LT Medial Points')
msv.PointCloudPlot(lt_CT.implant_roi_points_4d, 'LT Implant RoI Points')
lt_manipulator = msm.PointCloudManipulator(lt_CT.implant_roi_points_4d)
msv.PointCloudPlot(lt_manipulator.centred_nomalised_points,
                   'LT Implant Region (centred and normalised)')
centred_nomalised_thres_plot = msv.DensityThresholdPlot(
    lt_manipulator.centred_nomalised_points)
centred_nomalised_thres_plot.plot()
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
