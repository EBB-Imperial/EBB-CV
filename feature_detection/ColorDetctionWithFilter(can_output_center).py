import cv2
import numpy as np

# Load the image
image = cv2.imread('colourtest.jpg')

# Convert BGR to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define range of blue, red and yellow color in HSV
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])

lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 70, 50])
upper_red2 = np.array([180, 255, 255])

lower_yellow = np.array([20, 100, 100]) 
upper_yellow = np.array([40, 255, 255])

# Store center coordinates for each color
center_blue = []
center_red = []
center_yellow = []

def find_color(lower, upper, center_list, color_name, mask=None, min_area=10):
    # If no mask is provided, create one based on the color range
    if mask is None:
        mask = cv2.inRange(hsv, lower, upper)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image, image, mask=mask)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # Filter by area
        if cv2.contourArea(cnt) < min_area:
            continue  # Skip this contour

        # Calculate center of gravity
        M = cv2.moments(cnt)
        if M["m00"] != 0:  # to avoid division by zero
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Add the center to the color's list
            center_list.append((cX, cY))

            # Draw the contour and center of the shape on the image
            cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)
            image[cY, cX] = [255, 255, 255]  # Mark the center pixel white
            cv2.putText(image, color_name, (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


# Process for each color
find_color(lower_blue, upper_blue, center_blue, "blue")
find_color(lower_yellow, upper_yellow, center_yellow, "yellow")

# For red, we process the two ranges and combine the masks
mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask_red = cv2.bitwise_or(mask_red1, mask_red2)
find_color(None, None, center_red, "red", mask_red)

# Print the center coordinates
print("Blue centers: ", center_blue)
print("Red centers: ", center_red)
print("Yellow centers: ", center_yellow)

# Display the image
cv2.imshow("Image", image)
cv2.waitKey(0)
