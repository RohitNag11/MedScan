from medscan import (readers as msr,
                     viewers as msv,
                     segmenters as mss)
import numpy as np
import matplotlib.pyplot as plt

body_CT = msr.DicomCT(
    '/Users/rohit/Documents/Imperial/ME4/FYP/Sample Scans/MJM09_MJM010/MJM09_MJM010_Phantom1607,2003840n/R4G1B43W/TDS102WE')
lt_bone_mesh = msr.BoneMesh(
    '/Users/rohit/Documents/Imperial/ME4/FYP/Sample Scans/MJM09_MJM010/MJM09_2003840N_Left Tibia.stl', 'Left Tibia')
rt_bone_mesh = msr.BoneMesh(
    '/Users/rohit/Documents/Imperial/ME4/FYP/Sample Scans/MJM09_MJM010/MJM10_2003840N_Right Tibia.stl', 'Right Tibia')

# bone_meshes = [lt_bone_mesh, rt_bone_mesh]
# colors = ['#000000', '#FFFFFF']

# mesh_plot = msv.Bone3DPlot(bone_meshes, colors)
# mesh_plot.close()

# segemented_region_slider_plot = msv.SegmentedRegionSliderPlot(body_CT,
#                                                               bone_meshes,
#                                                               colors)
# segemented_region_slider_plot.close()


segmenter = mss.SoftTissueSegmenter(body_CT)
segmenter.add_bone_mesh(lt_bone_mesh)
segmenter.add_bone_mesh(rt_bone_mesh)

point_cloud_plot = msv.Density4DPlot(segmenter,
                                     rt_bone_mesh,
                                     350000,
                                     30,
                                     0.02,
                                     0,
                                     0.5)

lt_bone_segmented_point_cloud = segmenter.get_down_sampled_point_cloud(
    lt_bone_mesh,
    2)

# down_sampled_plot = msv.Density4DSliderPlot(lt_bone_segmented_point_cloud,
#                                             rt_bone_mesh,
#                                             100000,
#                                             30)
