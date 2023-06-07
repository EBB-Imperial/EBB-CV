import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('img/light_maze1.jpeg', cv2.IMREAD_GRAYSCALE)

lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
lap = np.uint8(np.absolute(lap))

sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)
sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))

sobelCombined = cv2.bitwise_or(sobelX, sobelY)

# sharpening
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
# sharpen sobelCombined
sharpened = cv2.filter2D(sobelCombined, -1, kernel)

titles = ['image', 'Laplacian', 'sobelX', 'sobelY', 'sobelCombined', 'sharpened']
images = [img, lap, sobelX, sobelY, sobelCombined, sharpened]

# Increase font size for titles
font_size = 16

for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i], fontsize=font_size)  # Set the fontsize for titles
    plt.xticks([]), plt.yticks([])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
