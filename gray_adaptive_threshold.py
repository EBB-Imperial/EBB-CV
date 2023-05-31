import cv2
import numpy as np

img = cv2.imread('./img/light_maze1.jpeg', 0)

# Calculate the target width or height while maintaining the aspect ratio
target_size = 1080
height, width = img.shape
if width > height:
    new_width = target_size
    new_height = int(height * (new_width / width))
else:
    new_height = target_size
    new_width = int(width * (new_height / height))

def make_thereshold(img):
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

closed_img = make_thereshold(img)

# Resize the window to match the image dimensions
cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Result', new_width, new_height)

while True:
    cv2.imshow("Result", closed_img)

    k = cv2.waitKey(1) & 0xFF

    if k == 27:  # Exit condition
        break

cv2.destroyAllWindows()
