import numpy as np
import cv2


def cartesian_2d_to_pixel_space(cartesian_points, x_bounds, y_bounds, ni, nj):
    '''takes a 2d array of points in cartesian space and translates them to pixel space'''
    x0, xn = x_bounds
    y0, yn = y_bounds
    cartesian_points[:, 0] -= x0
    cartesian_points[:, 1] -= yn
    T = np.array([[(ni - 1) / (xn - x0), 0],
                  [0, (nj - 1) / (y0 - yn)]])
    return np.int32(np.matmul(cartesian_points, T))


def get_poly_image(poly_pixels, ni, nj):
    '''Takes a 2d array of polygon edge points in pixel space and returns a binary image of the polygon'''
    img_dim = (ni, nj)
    img = np.zeros(img_dim)
    cv2.fillPoly(img,
                 pts=[poly_pixels],
                 color=255)
    return img / 255


def project_points_to_plane(points, normal, point_on_plane=np.array([0, 0, 0])):
    # Calculate the dot product of the normal vector and the difference between the points and a point on the plane
    d = np.dot(normal, (points - point_on_plane).T) / np.linalg.norm(normal)**2
    # Calculate the projection of the points onto the plane
    projections = points - np.outer(d, normal)
    return projections


def tetrahedron_volume(a, b, c, d):
    return np.abs(np.einsum('ij,ij->i', a-d, np.cross(b-d, c-d))) / 6


def translate_space_1d(cur_space_bounds, new_space_bounds, points):
    i0, i1 = cur_space_bounds
    x0, x1 = new_space_bounds
    return (points - i0) * (x1 - x0) / (i1 - i0) + x0


def translate_space_3d(point_cloud, cur_space_bounds_3d, new_space_bounds_3d):
    '''Arguments: 
    point_cloud: a 2d array of points in 3d space
    cur_space_bounds_3d: a 2d array of the bounds of the current space in the form [[x0, x1], [y0, y1], [z0, z1]]
    new_space_bounds_3d: a 2d array of the bounds of the new space in the form [[x0, x1], [y0, y1], [z0, z1]]
    '''
    # Calculate the translation and scaling factors
    cur_space_dim = cur_space_bounds_3d[:, 1] - cur_space_bounds_3d[:, 0]
    new_space_dim = new_space_bounds_3d[:, 1] - new_space_bounds_3d[:, 0]
    dx, dy, dz = new_space_bounds_3d[:, 0] - \
        cur_space_bounds_3d[:, 0] * (new_space_dim / cur_space_dim)
    kx, ky, kz = new_space_dim / cur_space_dim

    # Calculate the transformed point cloud
    transformed_point_cloud = (
        point_cloud * np.array([kx, ky, kz])) + np.array([dx, dy, dz])

    return transformed_point_cloud
