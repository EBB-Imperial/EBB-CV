import cv2
import numpy as np
import os

folder_path = "Picture"
images_list = os.listdir(folder_path)
print(f"Total number of images detected {len(images_list)}")

max_x = 0
max_y = 0

maze_size = (max_y, max_x)  # The size of the maze (height, width)
full_map = np.zeros((maze_size[0], maze_size[1], 3), dtype=np.uint8)  # RGB map

sift = cv2.SIFT_create()

# Assuming you have the centers of the images in terms of displacements from the first image center
image_centers = []  # This should be populated with your actual data

for img_file, center in zip(images_list, image_centers):
    img_path = os.path.join(folder_path, img_file) # Get the path to the image
    
    img = cv2.imread(img_path)  # Read as color
    img = cv2.resize(img, (0, 0), None, 0.5, 0.5) # Resize to half the size
        
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    full_map_gray = cv2.cvtColor(full_map, cv2.COLOR_BGR2GRAY)
        
    kp1, des1 = sift.detectAndCompute(img_gray, None) # Find keypoints and descriptors with SIFT
    kp2, des2 = sift.detectAndCompute(full_map_gray, None) # Find keypoints and descriptors with SIFT

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        height, width = full_map.shape[:2]
        img_warped = cv2.warpPerspective(img, H, (width, height))
        
        # Find position in full map to place the image based on the provided center
        x, y = center
        height, width = img_warped.shape[:2]

        # Correct the position in case the image is out of the bounds
        start_y = max(0, y - height // 2)
        end_y = min(y + height // 2, maze_size[0])
        start_x = max(0, x - width // 2)
        end_x = min(x + width // 2, maze_size[1])

        # Correct the portion of the image to place in the full map
        img_start_y = max(0, height // 2 - y)
        img_end_y = min(height, maze_size[0] + height // 2 - y)
        img_start_x = max(0, width // 2 - x)
        img_end_x = min(width, maze_size[1] + width // 2 - x)

        # Place the appropriate part of the image in the map
        full_map[start_y:end_y, start_x:end_x] = img_warped[img_start_y:img_end_y, img_start_x:img_end_x]

    cv2.imshow("Full Map", full_map)
    cv2.waitKey(1)

cv2.waitKey(0)
cv2.destroyAllWindows()

