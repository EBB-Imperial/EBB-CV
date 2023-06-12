import cv2
import numpy as np
import perspectivetest as transform
import combine
import os
import time
import sys
sys.path.append('Simulation_input')
import transform_getter as tg

def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    return rotated_image


map_width = 500
map_height = 800
piece_width = 200
piece_height = 200
angle = 0

true_map = combine.single_color_image(map_width, map_height, 255)

folder_path = "Simulation_input/2023.6.9_15.14"  # Replace with the actual folder path
filenames = os.listdir(folder_path)
transform_getter = tg.get_obj()
# Sort the filenames in ascending order
sorted_filenames = sorted(filenames)
# Print the sorted filenames

for filename in sorted_filenames:
    print(filename)
    if filename != "TransformRecord.txt":
        origin_piece, map_piece = transform.img_transform(folder_path + "/" + filename)
        filename = filename.replace('SampleScene_1080p_', '')
        filename = filename.replace('.jpg', '')
        filename = filename.replace('.png', '')
        new_record = transform_getter.get_transform(filename)
        x, _, y = new_record.position
        _, rotation, _ = new_record.rotation

        print(x, y)

        map_piece = cv2.resize(map_piece, (piece_width, piece_height))
        map_piece = rotate_image(map_piece, -rotation)
        true_map = combine.update_map(true_map, int(-y)*15, int(x)*11, map_piece)

cv2.imshow('Overall Map', true_map)
cv2.waitKey(0)
time.sleep(100)
    
 



# origin_piece, map_piece = transform.img_transform("Simulation_input/2023.6.9_15.14/SampleScene_1080p_09.06.2023_14-51-04.jpg")
# map_piece = cv2.resize(map_piece, (piece_width, piece_height))

# cv2.imshow('map_before rotate', map_piece)
# map_piece = rotate_image(map_piece, angle)

# true_map = combine.update_map(true_map, 10, 10, map_piece)

# cv2.imshow('map_piece', map_piece)
# cv2.imshow('Overall Map', true_map)

cv2.waitKey(0)
cv2.destroyAllWindows()


