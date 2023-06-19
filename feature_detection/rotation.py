import cv2
import numpy as np

def rotate_image(img, rotation_y):
    # Assuming rotation_y is the rotation angle around the y-axis
    # Convert the rotation angle from degrees to radians

    # Convert the image to RGBA
    img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

    # Get the size of the image
    height, width = img_rgba.shape[:2]
    
    # Compute the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rotation_y, 1)
    
    # Compute the new image dimensions
    abs_cos = abs(rotation_matrix[0,0])
    abs_sin = abs(rotation_matrix[0,1])
    
    new_width = int(height * abs_sin + width * abs_cos)
    new_height = int(height * abs_cos + width * abs_sin)
    
    # Adjust the rotation matrix to take into account the new image dimensions
    rotation_matrix[0, 2] += new_width / 2 - width / 2
    rotation_matrix[1, 2] += new_height / 2 - height / 2
    
    # Apply the rotation to the image
    img_rotated = cv2.warpAffine(img_rgba, rotation_matrix, (new_width, new_height), borderValue=(0, 0, 0, 0))
    
    return img_rotated
