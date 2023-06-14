import cv2
import numpy as np

def combine_images(img1, img2):
    # Convert the images to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGRA2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGRA2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Initialize BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Compute keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1_gray, None)
    kp2, des2 = orb.detectAndCompute(img2_gray, None)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract location of good matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp the image to match the combined image
    height, width = img1.shape[:2]
    img2_warped = cv2.warpPerspective(img2, H, (width, height))

    # Combine the warped image with the combined image
    combined_image = cv2.add(img1, img2_warped)

    return combined_image
