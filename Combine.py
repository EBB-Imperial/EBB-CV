import cv2
import numpy as np
import math

def single_color_image(width, height, color):
    # Create a blank black image
    map_image = np.zeros((int(height), int(width)), dtype=np.uint8)
    map_image[map_image == 0] = color
    return map_image


def create_maze_image(width, height):
    maze_image = np.zeros((height, width), dtype=np.uint8)
    maze_image[:, :] = 255
    cv2.rectangle(maze_image, (100, 100), (150, 200), 0, -1)
    cv2.rectangle(maze_image, (200, 200), (300, 300), 0, -1)
    cv2.rectangle(maze_image, (400, 400), (500, 500), 0, -1)

    return maze_image

def update_map(map_image, x, y, map_piece):

    x = math.ceil(x - map_piece.shape[1] / 2)
    y = math.ceil(y - map_piece.shape[0] / 2)

    # Get the dimensions of the piece of map
    piece_height, piece_width = map_piece.shape[:2]
    
    # Iterate over the pixels in the map piece
    for i in range(piece_height - 1):
        for j in range(piece_width - 1):
            
            if map_piece[i, j] >= 100:
                map_image[y+i, x+j] = 255
            
            elif map_piece[i, j] < 30 and map_image[y+i, x+j] < 100:
                map_image[y+i, x+j] = 0

    return map_image



# True_Map = create_maze_image(500,800)
# Cur_Map = single_color_image(200,200)
# cv2.imshow('True', True_Map)
# cv2.imshow('Cur', Cur_Map)
# cv2.waitKey(5000)
# cv2.destroyAllWindows()

# True_Map = update_map(True_Map, 100, 100, Cur_Map)
# # cv2.circle(True_Map, (10, 10), 2, (0, 0, 255), -1)
# cv2.imshow('True', True_Map)
# # cv2.imshow('Cur', Cur_Map)
# cv2.waitKey(10000)
# cv2.destroyAllWindows()



# while True:
    # x,y,map_piece = image_transform()
    # Cur_Map = update_map(Cur_Map, 0, 0, map_piece)
    # cv2.imshow('True', True_Map)
    # cv2.imshow('Cur', Cur_Map)