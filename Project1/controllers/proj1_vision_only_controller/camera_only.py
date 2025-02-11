import numpy as np
import math
from detect_circle import detect_circle

# Instructions:
# Review the `detect_circle` function to infer and detect a circle in the image and compute its angle.

def vision_only_distance_calculation(image, camera_fov, object_diameter):
    """
    This function performs object detection and calculates the depth and angle of the detected circle from a camera sensor.

    Args:
        image: The input image from the camera
        camera_fov: The field of view of the camera in radians.
        object_diameter: The expected diameter of the object in meters.

    Returns:
        depth: The depth to the detected object from camera depth estimation in meters.
        angle: the angle of the detected circle in radians.
    """

    ###########################################################################
    # TODO: Student code begins
    ###########################################################################
    
    #FoV = 0.84
    #Translation = 0.03, 0, 0.028
    #calling detect_circle
    detected = detect_circle(image)
    cx, cy, radius = detected[0]
    
    #image dimensions
    image_h, image_w = image.shape[:2]
    
    #focal length in pixels
    focal_px = image_w / (2 * math.tan(camera_fov/2))
    
    #diameter of the object based on detec_circle
    object_diameter_pixels = 2 * radius
    
    #depth calculation
    depth = (((focal_px) * object_diameter) / object_diameter_pixels)
    
    #angle calculation
    image_center_x = image_w / 2
    angle = ((cx - image_center_x) / image_w) * camera_fov
    ###########################################################################
    # Student code ends
    ###########################################################################

    return (depth + 0.03), angle