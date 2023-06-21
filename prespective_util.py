import cv2
import numpy as np
import time
import os
from scipy.signal import argrelextrema

empty_pixel_value = 66


def rotate_image(img, rotation_y, write_picture=False, rotation_pic_folder="rotation_pics"):
    # Assuming rotation_y is the rotation angle around the y-axis
    # Convert the rotation angle from degrees to radians

    # if image is empty, return empty image
    if img is None:
        return None
    if img.shape[0] == 0 or img.shape[1] == 0:
        return img

    # Convert the image to RGBA
    # img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    img_rgba = img

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
    img_rotated = cv2.warpAffine(img_rgba, rotation_matrix, (new_width, new_height), borderValue=empty_pixel_value)

    if write_picture:
        new_filename = rotation_pic_folder + time.strftime("/%Y-%m-%d_%H-%M-%S.png")
        for i in range(100):
            if os.path.exists(new_filename):
                new_filename = rotation_pic_folder + time.strftime("/%Y-%m-%d_%H-%M-%S") + "_" + str(i) + ".png"
            else:
                break

        cv2.imwrite(new_filename, img_rotated)
    
    return img_rotated


def perspective_transform(img, original_points, transformed_points, write_picture=False, perspective_transform_pic_folder="perspective_transform_pics"):
    # Convert lists to np.float32
    original_points = np.float32(original_points)
    transformed_points = np.float32(transformed_points)

    # Compute the perspective transform matrix
    perspective_transform_matrix = cv2.getPerspectiveTransform(original_points, transformed_points)

    # Find the maximum x and y coordinates for the width and height of the output image
    max_x = max([point[0] for point in transformed_points])
    max_y = max([point[1] for point in transformed_points])
    size = (int(max_x), int(max_y))

    # Apply the perspective transformation to the image
    transformed_img = cv2.warpPerspective(img, perspective_transform_matrix, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=empty_pixel_value)

    if write_picture:
        new_filename = perspective_transform_pic_folder + time.strftime("/%Y-%m-%d_%H-%M-%S.png")
        for i in range(100):
            if os.path.exists(new_filename):
                new_filename = perspective_transform_pic_folder + time.strftime("/%Y-%m-%d_%H-%M-%S") + "_" + str(i) + ".png"
            else:
                break

        cv2.imwrite(new_filename, transformed_img)
    
    return transformed_img



def make_threshold(img, write_picture=False, threshold_pic_folder="threshold_pics"):
    # Calculate histogram
    hist, bins = np.histogram(img.flatten(),256,[0,256])
    
    # Find local maxima 
    maxima = argrelextrema(hist, np.greater)[0]

    # Sort maxima by peak height
    sorted_maxima = sorted(maxima, key=lambda x: hist[x])
    
    # Get two highest peaks
    highest_peaks = sorted_maxima[-2:]
    
    # Calculate midpoint between the two highest peaks
    threshold_value = sum(highest_peaks) / 1.2
    
    # Apply global thresholding to convert the image to black and white
    _, binary_img = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
    
    if write_picture:
        new_filename = threshold_pic_folder + time.strftime("/%Y-%m-%d_%H-%M-%S.png")
        for i in range(100):
            if os.path.exists(new_filename):
                new_filename = threshold_pic_folder + time.strftime("/%Y-%m-%d_%H-%M-%S") + "_" + str(i) + ".png"
            else:
                break

        cv2.imwrite(new_filename, binary_img)
    
    return binary_img



# def make_threshold(img, write_picture=False, threshold_pic_folder="threshold_pics"):
#     # Apply adaptive thresholding to convert the image to black and white
#     _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

#     # Perform morphological closing to fill in the gaps
#     kernel_size = 8
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

#     # Perform morphological opening to remove noise
#     binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=1)
#     closed_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=1)

#     if write_picture:
#         new_filename = threshold_pic_folder + time.strftime("/%Y-%m-%d_%H-%M-%S.png")
#         for i in range(100):
#             if os.path.exists(new_filename):
#                 new_filename = threshold_pic_folder + time.strftime("/%Y-%m-%d_%H-%M-%S") + "_" + str(i) + ".png"
#             else:
#                 break

#         cv2.imwrite(new_filename, closed_img)

#     return closed_img


def downsampling(img, ratio):
    # Downsample the image
    height, width = img.shape[:2]
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    downsized_img = cv2.resize(img, (new_width, new_height))

    return downsized_img