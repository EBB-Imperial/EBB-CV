import cv2
import numpy as np
import perspectivetest as transform
import combine


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
angle = -45

true_map = combine.single_color_image(map_width, map_height, 255)

origin_piece, map_piece = transform.img_transform("img/light_maze1.png")

map_piece = cv2.resize(map_piece, (piece_width, piece_height))

cv2.imshow('map_before rotate', map_piece)
map_piece = rotate_image(map_piece, angle)

true_map = combine.update_map(true_map, 10, 10, map_piece)

cv2.imshow('map_piece', map_piece)
cv2.imshow('Overall Map', true_map)

cv2.waitKey(0)
cv2.destroyAllWindows()


