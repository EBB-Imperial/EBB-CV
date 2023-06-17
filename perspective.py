import cv2
import numpy as np

def perspective_transform(img):
    height, width = img.shape[:2]
    original_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    transformed_points = np.float32([[0, 0], [width, 0], [-5*width, 1.5* height], [6.2*width, 1.5 * height]])
    matrix = cv2.getPerspectiveTransform(transformed_points, original_points)
    h, _ = cv2.findHomography(original_points, transformed_points)
    transformed_img = cv2.warpPerspective(img, h, (2 * width, 2 * height))
    return transformed_img

# if __name__ == "__main__":  

#     # Example usage
#     image = cv2.imread("/home/yi/Documents/EBB-CV/SimData_2023-6-13/SampleScene_1080p_13.06.2023_20-01-12.jpg")

#     # Set the camera parameters
#     camera_height = 5.0  # meters
#     fov = 60.0  # degrees
#     camera_angle = 30.0  # degrees
#     output_resolution = (800, 600)

#     # Perform IPM
#     result = ipm(image, camera_height, fov, camera_angle, output_resolution)

#     # Display the result
#     cv2.imshow("IPM Result", result)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
