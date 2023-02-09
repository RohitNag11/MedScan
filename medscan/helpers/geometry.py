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
                 color=(255, 255, 255))
    return img


def project_points_to_plane(points, normal, point_on_plane=np.array([0, 0, 0])):
    # Calculate the dot product of the normal vector and the difference between the points and a point on the plane
    d = np.dot(normal, (points - point_on_plane).T) / np.linalg.norm(normal)**2
    # Calculate the projection of the points onto the plane
    projections = points - np.outer(d, normal)
    return projections
