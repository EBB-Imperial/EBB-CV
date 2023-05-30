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
def create_maze_image(width, height):
    # Create a blank black image
    maze_image = np.zeros((height, width), dtype=np.uint8)

    # Set the walls as white
    maze_image[:, :] = 255

    # Draw the maze pattern
    # Adjust the coordinates and sizes to customize the maze design
    cv2.rectangle(maze_image, (100, 100), (150, 200), 0, -1)
    cv2.rectangle(maze_image, (200, 200), (300, 300), 0, -1)
    cv2.rectangle(maze_image, (400, 400), (500, 500), 0, -1)
    # Add more maze patterns as needed...

    return maze_image

# Map = create_map(2410,3670)
# map_piece = create_map_black(300,300)
# Map = update_map(Map, 100, 100, map_piece)
Map = create_maze_image(500,800)

cv2.imshow('Image', Map)
cv2.waitKey(10000)
cv2.destroyAllWindows()