import math
import numpy as np
from detect_circle import detect_circle

# Instructions:
# Step 1: Review the `detect_circle` function to infer and detect a circle in the image 
#         and compute its angle.
# Step 2: Explore the LiDAR data corresponding to the detected object. Investigate 
#         how to classify the object as either a sphere or a disk. You may reuse 
#         your code from `camera_only.py`.

def camera_and_lidar_calculation(image, camera_fov, object_diameter, lidar_data):
    """
    Performs object detection and classification by combining data from a camera and a LiDAR sensor.

    Args:
        image: The input image captured by the camera.
        camera_fov: The field of view of the camera in radians.
        object_diameter: The expected diameter of the object in meters.
        lidar_data: An array representing LiDAR distance data indexed by angle 
                                  in degrees, where each element corresponds to the distance 
                                  at a specific angle.

    Returns:
        lidar_distance: The distance to the detected object from the LiDAR sensor in meters.
        object_shape: A string indicating the shape of the detected object ("sphere" or "disk").
    """

    ###########################################################################
    # TODO: Student code begins
    ###########################################################################
    
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
    depth = ((focal_px * object_diameter) / object_diameter_pixels)
    
    #angle calculation
    image_center_x = image_w / 2
    angle = ((cx - image_center_x) / image_w) * camera_fov
    angle_deg = int(math.degrees(angle)) % 360
    
    #distance calculation between centers
    lidar_distance = lidar_data[angle_deg]
    
    #classifying as sphere
    # angle_range = int(math.degrees(camera_fov/2))
    
    angle_range = 5
    lidar_distances = []
    for i in range(angle_deg - angle_range, angle_deg + angle_range + 1):
        if (lidar_data[i%360]-lidar_distance <= (object_diameter/2)):
            lidar_distances.append(lidar_data[i % 360])
        print(lidar_data[i%360])
    distance_std = np.std(lidar_distances)
    print(distance_std)
    if distance_std > 0.002:
        object_shape="sphere"
    else:
        object_shape="disk"
        
        
    ###########################################################################
    # Student code ends
    ###########################################################################

    return lidar_distance, object_shape
