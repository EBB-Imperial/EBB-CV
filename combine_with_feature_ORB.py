import cv2
import numpy as np
import os

folder_path = "Picture" # The folder with the images
images_list = [f for f in os.listdir(folder_path) if f.endswith('.png')] # Only load .png files
print(f"Total number of images detected {len(images_list)}") # Print the number of images detected

max_x = 0 # These are the maximum x coordinates of the images
max_y = 0 # These are the maximum y coordinates of the images

maze_size = (max_y, max_x)  # The size of the maze (height, width)
full_map = np.zeros((maze_size[0], maze_size[1], 3), dtype=np.uint8)  # RGB map

orb = cv2.ORB_create()

# Assuming you have the centers of the images in terms of displacements from the first image center
image_centers = []  # This should be populated with your actual data

for img_file, center in zip(images_list, image_centers):
    img_path = os.path.join(folder_path, img_file) # Get the path to the image
    
    img = cv2.imread(img_path)  # Read as color
    img = cv2.resize(img, (0, 0), None, 0.5, 0.5) # Resize to half the size
        
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to grayscale
    full_map_gray = cv2.cvtColor(full_map, cv2.COLOR_BGR2GRAY) # Convert to grayscale
        
    kp1, des1 = orb.detectAndCompute(img_gray, None) # Find keypoints and descriptors
    kp2, des2 = orb.detectAndCompute(full_map_gray, None) # Find keypoints and descriptors

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # Create BFMatcher object
    matches = bf.match(des1, des2) # Match descriptors

    matches = sorted(matches, key=lambda x: x.distance) # Sort them in the order of their distance

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2) # Extract location of good matches
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2) # Extract location of good matches

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) # Find homography
    height, width = full_map.shape[:2] # Get shape of full map
    img_warped = cv2.warpPerspective(img, H, (width, height)) # Warp source image to destination based on homography
        
    # Find position in full map to place the image based on the provided center
    x, y = center # The center of the image
    height, width = img_warped.shape[:2] # The size of the image

    # Correct the position in case the image is out of the bounds
    start_y = max(0, y - height // 2) # The top left corner of the image
    end_y = min(y + height // 2, maze_size[0]) # The bottom right corner of the image
    start_x = max(0, x - width // 2) # The top left corner of the image
    end_x = min(x + width // 2, maze_size[1]) # The bottom right corner of the image

    # Correct the portion of the image to place in the full map
    img_start_y = max(0, height // 2 - y) # The top left corner of the image
    img_end_y = min(height, maze_size[0] + height // 2 - y) # The bottom right corner of the image
    img_start_x = max(0, width // 2 - x) # The top left corner of the image
    img_end_x = min(width, maze_size[1] + width // 2 - x) # The bottom right corner of the image
 
    # Place the appropriate part of the image in the map
    full_map[start_y:end_y, start_x:end_x] = img_warped[img_start_y:img_end_y, img_start_x:img_end_x] # Place the image in the full map

    cv2.imshow("Full Map", full_map) # Show the full map
    cv2.waitKey(1) # Wait for a key press

cv2.waitKey(0) # Wait for a key press
cv2.destroyAllWindows() # Close all windows
