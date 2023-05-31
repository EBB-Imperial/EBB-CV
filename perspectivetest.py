import cv2
import numpy as np

# Load image
# img = cv2.imread("img/light_maze1.png", 0)

# # Calculate the target width or height while maintaining the aspect ratio
# target_size = 1080
# height, width = img.shape

# if width > height:
#     new_width = target_size
#     new_height = int(height * (new_width / width))
# else:
#     new_height = target_size
#     new_width = int(width * (new_height / height))
def img_transform(img_src):
    img = cv2.imread(img_src, 0)

    # Calculate the target width or height while maintaining the aspect ratio
    target_size = 1080
    height, width = img.shape

    if width > height:
        new_width = target_size
        new_height = int(height * (new_width / width))
    else:
        new_height = target_size
        new_width = int(width * (new_height / height))

    result1 = make_threshold(img, new_width, new_height)
    original_points = np.float32([[0, 0], [new_width, 0], [0, new_height], [new_width, new_height]])
    transformed_points = np.float32([[0, 0], [new_width, 0], [-new_width, 1.5* new_height], [2*new_width, 1.5 * new_height]])
    matrix = cv2.getPerspectiveTransform(transformed_points, original_points)
    result2 = cv2.warpPerspective(result1, matrix, (new_width, new_height))

    return result1, result2

def make_threshold(img, new_width, new_height):
    # Resize the image with the calculated dimensions
    resized_img = cv2.resize(img, (new_width, new_height))

    # Apply adaptive thresholding to convert the image to black and white
    _, binary_img = cv2.threshold(resized_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Perform morphological closing to fill in the gaps
    kernel_size = 8
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # Perform morphological opening to remove noise
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=1)
    closed_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=1)

    return closed_img

# result1 = make_threshold(img)

# # Resize the window to match the image dimensions
# cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Result', new_width, new_height)

# # Perform perspective transform
# original_points = np.float32([[0, 0], [new_width, 0], [0, new_height], [new_width, new_height]])
# transformed_points = np.float32([[0, 0], [new_width, 0], [-new_width, 1.5* new_height], [2*new_width, 1.5 * new_height]])
# matrix = cv2.getPerspectiveTransform(transformed_points, original_points)
# result2 = cv2.warpPerspective(result1, matrix, (new_width, new_height))


if (0):
    result1, result2 = img_transform("img/light_maze1.png")
    # Display the image
    cv2.imshow('Result', result1)
    cv2.imshow('Result2', result2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()









