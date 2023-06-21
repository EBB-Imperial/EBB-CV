import cv2
import numpy as np
import perspectivetest as transform
import Combine as combine
import os
import time
import json
from transform_getter import TransformGetter
from prespective_util import rotate_image, perspective_transform, make_threshold, downsampling


DOWNSAMPLE_RATIO = 0.05
rotation_pic_folder = "rotation_pics"
perspective_transform_pic_folder = "perspective_transform_pics"
threshold_transform_pic_folder = "threshold_pics"
empty_pixel_value = 66
output_data_txt = "output_data.txt"


def main():

    folder_path = "SimData_2023-6-13"
    filenames = os.listdir(folder_path)
    transform_getter = TransformGetter(folder_path + "/EEB_Transform.txt")
    # Sort the filenames in ascending order
    sorted_filenames = sorted(filenames)

    map_output_path = "map_output"

    # if no map_output folder, create one
    if not os.path.exists(map_output_path):
        os.mkdir(map_output_path)

    # if no rotation_pics folder, create one
    if not os.path.exists(rotation_pic_folder):
        os.mkdir(rotation_pic_folder)

    # if no perspective_transform_pics folder, create one
    if not os.path.exists(perspective_transform_pic_folder):
        os.mkdir(perspective_transform_pic_folder)

    # if no threshold_pics folder, create one
    if not os.path.exists(threshold_transform_pic_folder):
        os.mkdir(threshold_transform_pic_folder)

    # clear the map_output folder
    for filename in os.listdir(map_output_path):
        os.remove(map_output_path + "/" + filename)
    
    # clear the rotation_pics folder and perspective_transform_pics folder
    for filename in os.listdir(rotation_pic_folder):
        os.remove(rotation_pic_folder + "/" + filename)
    for filename in os.listdir(perspective_transform_pic_folder):
        os.remove(perspective_transform_pic_folder + "/" + filename)
    for filename in os.listdir(threshold_transform_pic_folder):
        os.remove(threshold_transform_pic_folder + "/" + filename)
    
    # clear the output_data.txt
    with open(output_data_txt, "w") as f:
        f.write("")

    
    map_offset_x = 0        # positions of courners can have negative values, so we need to offset the map
    map_offset_y = 0
    map_size_x = 0
    map_size_y = 0

    # used to store the original coordinates of the map_piece
    original_x = original_y = 0

    true_map = combine.single_color_image(0, 0, empty_pixel_value)
    last_image = true_map
    first_image = True

    for filename in sorted_filenames:

        if filename != "EEB_Transform.txt":

            print("processing " + filename + "...")
            map_piece = cv2.imread(folder_path + "/" + filename, 0)
            map_piece = downsampling(map_piece, DOWNSAMPLE_RATIO)

            img_height, img_width = map_piece.shape[:2]

            timestamp = filename.replace('SampleScene_1080p_', '').removesuffix('.png').removesuffix('.jpg')
            new_record = transform_getter.get_transform(timestamp)
            x, _, y = new_record.position
            _, rotation, _ = new_record.rotation
            original_x = x
            original_y = y

            # preprocess image
            map_piece = make_threshold(map_piece, write_picture=True, threshold_pic_folder=threshold_transform_pic_folder)
            original_points = np.float32([[0, 0], [img_width, 0], [0, img_height], [img_width, img_height]])
            transformed_points = np.float32([[0, 0], [img_width * 2.991, 0], [img_width * 0.991, img_width * 1.358], [img_width * 1.991, img_width * 1.358]])
            # transformed_points = np.float32([[0, 0], [img_width, 0], [0, img_height], [img_width, img_height]])
            # original_points = np.float32([[0, 0], [img_width, 0], [-img_width  * 0.991, img_width * 1.358], [img_width * 1.991, img_width * 1.358]])
            map_piece = perspective_transform(map_piece, original_points, transformed_points, write_picture=True, perspective_transform_pic_folder=perspective_transform_pic_folder)
            map_piece = rotate_image(map_piece, rotation_y=-rotation, write_picture=True, rotation_pic_folder=rotation_pic_folder)

            # update the width and height of the map_piece after rotation
            rotated_img_height, rotated_img_width = map_piece.shape[:2]

            # calculate the position of the map_piece
            # positions of four corners of camera on 2d plane:
            # (-31.29, 0.00, 13.83)，(-23.32, 0.00, 3.01)，(-46.78, 0.00, 3.01)，(-38.81, 0.00, 13.83)
            # position of camera is (-35.05, 0.3.68999, 13.91)
            near_edge_distance = 13.91 - 13.83
            far_edge_distance = 13.91 - 3.01
            center_distance = (far_edge_distance + near_edge_distance) / 2
            d_to_pixel_ratio = img_width / (31.29 - 23.32)
            x = -(x + center_distance * np.sin(-rotation * np.pi / 180)) * d_to_pixel_ratio
            y = (y - center_distance * np.cos(rotation * np.pi / 180)) * d_to_pixel_ratio

            # calculate the offset of the map_piece
            piece_min_x = x - rotated_img_width / 2
            piece_min_y = y - rotated_img_height / 2
            piece_max_x = x + rotated_img_width / 2
            piece_max_y = y + rotated_img_height / 2

            # update the offset and size of the true_map dynamically
            if not first_image:
                new_map_offset_x = -min(-map_offset_x, piece_min_x)
                new_map_offset_y = -min(-map_offset_y, piece_min_y)
            else:
                new_map_offset_x = -piece_min_x
                new_map_offset_y = -piece_min_y
            
            delta_x = new_map_offset_x - map_offset_x
            delta_y = new_map_offset_y - map_offset_y
            map_offset_x = new_map_offset_x
            map_offset_y = new_map_offset_y
            map_size_x = max(map_size_x, piece_max_x + map_offset_x)
            map_size_y = max(map_size_y, piece_max_y + map_offset_y)

            # create a new blank map with the updated size
            if not first_image:
                blank_canvas = combine.single_color_image(map_size_x + delta_x, map_size_y + delta_y, empty_pixel_value)
                map_size_x += delta_x
                map_size_y += delta_y
            else:
                blank_canvas = combine.single_color_image(map_size_x, map_size_y, empty_pixel_value)
                first_image = False

            # combine the last image and blank canvas
            last_image = combine.update_map(blank_canvas, delta_x + last_image.shape[1] / 2, delta_y + last_image.shape[0] / 2, last_image)

            # output last image
            cv2.imwrite(map_output_path + "/" + "map_" + filename + ".png", last_image)
            
            x += map_offset_x
            y += map_offset_y

            print(x, y, rotation)

            # combine the map_piece and last_image
            last_image = combine.update_map(last_image, x, y, map_piece)

            # before exporting the map, write the data to output_data.txt
            with open(output_data_txt, "a") as f:
                # write in json: ImageName: x, y, rotation, d_to_pixel_ratio, map_offset_x, map_offset_y, map_size_x, map_size_y
                data = {filename: [original_x, original_y, rotation, d_to_pixel_ratio, map_offset_x, map_offset_y, map_size_x, map_size_y]}
                json.dump(data, f)
                f.write("\n")

            # export the map
            cv2.imwrite(map_output_path + "/" + "map_" + filename + ".png", last_image)

    # cv2.imshow('Overall Map', true_map)
    # cv2.waitKey(0)
    # time.sleep(100)

    print("process finished")
        
    



    # origin_piece, map_piece = transform.img_transform("Simulation_input/2023.6.9_15.14/SampleScene_1080p_09.06.2023_14-51-04.jpg")
    # map_piece = cv2.resize(map_piece, (piece_width, piece_height))

    # cv2.imshow('map_before rotate', map_piece)
    # map_piece = rotate_image(map_piece, angle)

    # true_map = combine.update_map(true_map, 10, 10, map_piece)

    # cv2.imshow('map_piece', map_piece)
    # cv2.imshow('Overall Map', true_map)


if __name__ == "__main__":
    main()
