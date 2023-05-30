import cv2
import numpy as np

# def AddMap (new_map, rob_x, rob_y, xy_angle, z_angle, Map):
#     map_x = rob_x + 100
#     map_y = rob_y + 100
def create_map(width, height):
    # Create a blank black image
    map_image = np.zeros((height, width), dtype=np.uint8)
    # map_image[map_image == 0] = 255
    return map_image

def create_map_black(width, height):
    # Create a blank black image
    map_image = np.zeros((height, width), dtype=np.uint8)
    map_image[map_image == 0] = 255
    return map_image

def update_map(map_image, x, y, piece_of_map):
    # Get the dimensions of the piece of map
    piece_height, piece_width = piece_of_map.shape[:2]

    # Update the corresponding region in the map image with the piece of map
    map_image[y:y+piece_height, x:x+piece_width] = piece_of_map

    return map_image

Map = create_map(2410,3670)
map_piece = create_map_black(300,300)
Map = update_map(Map, 100, 100, map_piece)

cv2.imshow('Image', Map)
cv2.waitKey(10000)
cv2.destroyAllWindows()