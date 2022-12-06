from medscan import (readers as msr,
                     viewers as msv)

body_CT = msr.DicomCT(
    '/Users/rohit/Documents/Imperial/ME4/FYP/Sample Scans/MJM09_MJM010/MJM09_MJM010_Phantom1607,2003840n/R4G1B43W/TDS102WE')
lt_bone_mesh = msr.BoneMesh(
    '/Users/rohit/Documents/Imperial/ME4/FYP/Sample Scans/MJM09_MJM010/MJM09_2003840N_Left Tibia.stl')
rt_bone_mesh = msr.BoneMesh(
    '/Users/rohit/Documents/Imperial/ME4/FYP/Sample Scans/MJM09_MJM010/MJM10_2003840N_Right Tibia.stl')

bone_meshes = [lt_bone_mesh, rt_bone_mesh]
labels = ['Left Tibia', 'Right Tibia']
colors = ['#4DFF00', '#00D9FF']

mesh_plot = msv.Bone3DPlot(bone_meshes, labels, colors)
mesh_plot.close()

segemented_slider_plot = msv.SegmentedSliderPlot(body_CT,
                                                 bone_meshes,
                                                 labels,
                                                 colors)
segemented_slider_plot.close()
