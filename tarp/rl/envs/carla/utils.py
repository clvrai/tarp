import os, sys
import numpy as np

import carla

def distance_to_line(A, B, p):
    num   = np.linalg.norm(np.cross(B - A, A - p))
    denom = np.linalg.norm(B - A)
    if np.isclose(denom, 0):
        return np.linalg.norm(p - A)
    return num / denom

def vector(v):
    """ Turn carla Location/Vector3D/Rotation to np.array """
    if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
        return np.array([v.x, v.y, v.z])
    elif isinstance(v, carla.Rotation):
        return np.array([v.pitch, v.yaw, v.roll])

def angle_diff(v0, v1):
    """ Calculates the signed angle difference (-pi, pi] between 2D vector v0 and v1 """
    angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
    if angle > np.pi: angle -= 2 * np.pi
    elif angle <= -np.pi: angle += 2 * np.pi
    return angle

def to_bgra_array(image):
    """Convert a CARLA raw image to a BGRA numpy array."""
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    return array


def to_rgb_array(image):
    """Convert a CARLA raw image to a RGB numpy array."""
    array = to_bgra_array(image)
    # Convert BGRA to RGB.
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array

def labels_to_array(image):
    """
    Convert an image containing CARLA semantic segmentation labels to a 2D array
    containing the label of each pixel.
    """
    return to_bgra_array(image)[:, :, 2]

def labels_to_cityscapes_palette(image, specified_classes=None):
    """
    Convert an image containing CARLA semantic segmentation labels to
    Cityscapes palette.
    """
    classes = {
        0: [0, 0, 0],         # None
        1: [70, 70, 70],      # Buildings
        2: [100, 40, 40],   # Fences
        3: [55, 90, 80],       # Other
        4: [220, 20, 60],     # Pedestrians
        5: [153, 153, 153],   # Poles
        6: [157, 234, 50],    # RoadLines
        7: [128, 64, 128],    # Roads
        8: [244, 35, 232],    # Sidewalks
        9: [107, 142, 35],    # Vegetation
        10: [0, 0, 142],      # Vehicles
        11: [102, 102, 156],  # Walls
        12: [220, 220, 0],     # TrafficSigns
        13: [70, 130, 180],     # Sky
        14: [81, 0, 81],     # Ground
        15: [150, 100, 100],     # Bridge
        16: [230, 150, 140],     # RailTrack
        17: [180, 165, 180],     # GuardRail
        18: [250, 170, 30],     # TrafficLight
        19: [110, 190, 160],     # Static
        20: [170, 120, 50],     # Dynamic
        21: [45, 60, 150],     # Water
        22: [145, 170, 100],     # Terrain
    }
    array = labels_to_array(image)
    result = np.zeros((array.shape[0], array.shape[1], 3))
    if specified_classes is None:
        for key, value in classes.items():
            result[np.where(array == key)] = value
    else:
        for class_id in specified_classes:
            result[np.where(array == class_id)] = classes[class_id]
    return result
