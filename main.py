from medscan import (readers as msr,
                     viewers as msv,
                     segmenters as mss)

body_CT = msr.DicomCT(
    '/Users/rohit/Documents/Imperial/ME4/FYP/Sample Scans/MJM09_MJM010/MJM09_MJM010_Phantom1607,2003840n/R4G1B43W/TDS102WE')
lt_bone_mesh = msr.BoneMesh(
    '/Users/rohit/Documents/Imperial/ME4/FYP/Sample Scans/MJM09_MJM010/MJM09_2003840N_Left Tibia.stl', 'Left Tibia')
rt_bone_mesh = msr.BoneMesh(
    '/Users/rohit/Documents/Imperial/ME4/FYP/Sample Scans/MJM09_MJM010/MJM10_2003840N_Right Tibia.stl', 'Right Tibia')

segmenter = mss.SoftTissueSegmenter(body_CT)
segmenter.addBoneMeshSlices(lt_bone_mesh)

print(segmenter.raw_slices)
print(segmenter.segmented_slices[lt_bone_mesh.name])


# bone_meshes = [lt_bone_mesh, rt_bone_mesh]
# colors = ['#4DFF00', '#00D9FF']

# mesh_plot = msv.Bone3DPlot(bone_meshes, colors)
# mesh_plot.close()

# segemented_region_slider_plot = msv.SegmentedRegionSliderPlot(body_CT,
#                                                  bone_meshes,
#                                                  colors)
# segemented_slider_plot.close()
