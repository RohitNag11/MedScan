import matplotlib
from mpl_toolkits import mplot3d
from stl.mesh import Mesh
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ss = 0.6
origin = np.array([-190.630859375, -341.130859375, -84])

lt_m = Mesh.from_file(
    '/Users/rohit/Documents/Imperial/ME4/FYP/Sample Scans/MJM09_MJM010/MJM09_2003840N_Left Tibia.stl')
rt_m = Mesh.from_file(
    '/Users/rohit/Documents/Imperial/ME4/FYP/Sample Scans/MJM09_MJM010/MJM10_2003840N_Right Tibia.stl')
lt_min_z, lt_maz_z = np.min(lt_m.z), np.max(lt_m.z)
rt_min_z, rt_maz_z = np.min(rt_m.z), np.max(rt_m.z)
print(f'lt_min_z, lt_max_z: {lt_min_z, lt_maz_z}')
print(f'rt_min_z, rt_max_z: {lt_min_z, lt_maz_z}')
lt_height = lt_maz_z - lt_min_z
rt_height = rt_maz_z - rt_min_z
lt_no_slices = np.floor(lt_height / ss) + 1
rt_no_slices = np.floor(rt_height / ss) + 1
print(f'lt_no_slices: {lt_no_slices}')
print(f'rt_no_slices: {rt_no_slices}')
lt_m.translate(origin * -1)
rt_m.translate(origin * -1)
lt_knee_loc = np.max(lt_m.z)
rt_knee_loc = np.max(rt_m.z)
print(f'lt_knee_loc: {lt_knee_loc}')
print(f'rt_knee_loc: {rt_knee_loc}')
ax.add_collection3d(
    mplot3d.art3d.Poly3DCollection(
        lt_m.vectors,
        edgecolor='#0015FF16',
        facecolor='#0015FF15',
        alpha=0.4,
        label='left tibia')
)
ax.add_collection3d(
    mplot3d.art3d.Poly3DCollection(
        rt_m.vectors,
        edgecolor='#FF00001C',
        facecolor='#FF000012',
        alpha=0.4,
        label='right tibia')
)
scale = lt_m.points.flatten()
ax.auto_scale_xyz(scale, scale, scale)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
