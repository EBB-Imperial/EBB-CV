import cv2
import numpy as np

def ipm(image, camera_height, fov, camera_angle, output_resolution):
    # Step 1: Calculate camera parameters
    image_width, image_height = image.shape[1], image.shape[0]
    center_x, center_y = image_width / 2, image_height / 2

    # Calculate the distance to the ground plane
    d = camera_height / np.tan(np.radians(camera_angle))

    # Step 2: Determine image coordinates
    image_coords = np.indices((image_height, image_width))
    x_image, y_image = image_coords[1] - center_x, image_coords[0] - center_y

    # Step 3: Perform IPM transformation
    output_width, output_height = output_resolution
    output_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)

    for x_out in range(output_width):
        for y_out in range(output_height):
            # Convert output pixel coordinates to ground plane coordinates
            x_ground = (x_out - output_width / 2) * d / output_width
            y_ground = (y_out - output_height / 2) * d / output_height

            # Calculate the corresponding horizontal and vertical angles
            alpha = np.arctan2(x_ground, d)
            beta = np.arctan2(y_ground, np.sqrt(x_ground**2 + d**2))

            # Calculate the distance from the camera to the pixel on the ground plane
            r = d / np.cos(beta)

            # Calculate the corresponding pixel coordinates in the input image
            x_in = int(center_x + (alpha * np.degrees(fov) * image_width) / 360)
            y_in = int(center_y + (beta * np.degrees(fov) * image_height) / 360)

            if 0 <= x_in < image_width and 0 <= y_in < image_height:
                output_image[y_out, x_out] = image[y_in, x_in]

    return output_image

# Example usage
image = cv2.imread("/home/yi/Documents/EBB-CV/SimData_2023-6-13/SampleScene_1080p_13.06.2023_20-01-12.jpg")

# Set the camera parameters
camera_height = 5.0  # meters
fov = 60.0  # degrees
camera_angle = 30.0  # degrees
output_resolution = (800, 600)

# Perform IPM
result = ipm(image, camera_height, fov, camera_angle, output_resolution)

# Display the result
cv2.imshow("IPM Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
