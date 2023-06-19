import cv2
import numpy as np
import sys
from PIL import Image

def detectAndDescribe(image):
    # 建立SIFT生成器
    sift = cv2.SIFT_create()
    # 检测SIFT特征点，并计算描述子
    (kps, features) = sift.detectAndCompute(image, None)
    # 将结果转换成NumPy数组
    kps = np.float32([kp.pt for kp in kps])
    # 返回特征点集，及对应的描述特征
    return (kps, features)

def image_stitching(imageA,imageB):

    #检测A、B图片的SIFT关键特征点，并计算特征描述子
    kpsA, featuresA = detectAndDescribe(imageA)
    kpsB, featuresB = detectAndDescribe(imageB)
    # 建立暴力匹配器
    bf = cv2.BFMatcher()
    # 使用KNN检测来自A、B图的SIFT特征匹配对，K=2
    matches = bf.knnMatch(featuresA, featuresB, 2)
    good = []
    for m in matches:
        # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
        if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
            # 存储两个点在featuresA, featuresB中的索引值
            good.append((m[0].trainIdx, m[0].queryIdx))

    if len(good) > 4:
        # 获取匹配对的点坐标
        ptsA = np.float32([kpsA[i] for (_, i) in good])
        ptsB = np.float32([kpsB[i] for (i, _) in good])
        # 计算视角变换矩阵
        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,4.0)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), 4))
        return None

    if H is None:
        print("Homography matrix is None")
        return None

    # Calculate the size of the new image
    h1, w1 = imageA.shape[:2]
    h2, w2 = imageB.shape[:2]
    corners = np.array([
        [0, 0],
        [0, h1],
        [w1, h1],
        [w1, 0]
    ], dtype=np.float32)
    corners = np.array([corners])
    warped_corners = cv2.perspectiveTransform(corners, H)
    warped_corners = np.round(warped_corners).astype(int)
    x_start = min(warped_corners[0,:,0])
    y_start = min(warped_corners[0,:,1])
    x_end = max(warped_corners[0,:,0])
    y_end = max(warped_corners[0,:,1])

    width = max(x_end, w2) - min(x_start, 0)
    height = max(y_end, h2) - min(y_start, 0)

    # Create a translation matrix to shift the image
    M = np.array([
        [1, 0, -min(x_start, 0)],
        [0, 1, -min(y_start, 0)],
        [0, 0, 1]
    ], dtype=np.float32)

    # Apply the translation and homography transformation to imageA
    warped_imageA = cv2.warpPerspective(imageA, M.dot(H), (width, height))
    
    # Apply the translation transformation to imageB
    warped_imageB = cv2.warpAffine(imageB, M[:2], (width, height))
    
    # Create the output image and apply a mask to remove black borders
    output = warped_imageA.copy()
    mask = (warped_imageB.sum(axis=2) > 0)
    output[mask] = warped_imageB[mask]

    return output

if __name__ == "__main__":
    import os
    import shutil
    import cv2

    # Define the image directory
    combined_dir = 'existing_combined_images'

    # Delete the combined images directory if it exists
    if os.path.exists(combined_dir):
        shutil.rmtree(combined_dir)

    # Create the combined images directory
    os.makedirs(combined_dir)

    # Directory containing the images
    image_dir = 'combine_screenshot'

    # Create a set to store the names of processed images
    processed_images = set()

    # Get a list of filenames and sort it
    filenames = os.listdir(image_dir)
    filenames = list(filter(lambda filename: filename.endswith('.png'), filenames))
    filenames.sort()

    print(filenames)

    # set the first image as the base image
    first_image_filename = filenames[0]
    first_image_paths = os.path.join(image_dir, first_image_filename)
    first_image = cv2.imread(first_image_paths)

    # Save the first image as the combined image
    combined_image_filename = 'existing_map_1.png'
    cv2.imwrite(os.path.join(combined_dir, combined_image_filename), first_image)

    # Add the first image to the set of processed images
    processed_images.add(first_image_filename)

    while True:
        for filename in filenames:
            # Check if the file is a .jpg image and has not been processed
            if filename.endswith('.png') and filename not in processed_images:

                # get the image path
                image_path = os.path.join(image_dir, filename)

                # get the combined image path
                combined_image_path = os.path.join(combined_dir, combined_image_filename)

                img_name = [combined_image_path, image_path]

                # stitch the images
                combined_image = image_stitching(cv2.imread(combined_image_path), cv2.imread(image_path))

                if combined_image is not None:
                    # Save the combined image
                    combined_image_filename = 'existing_map_' + str(len(processed_images) + 1) + '.png'
                    cv2.imwrite(os.path.join(combined_dir, combined_image_filename), combined_image)

                    # Add the filename to the set of processed images
                    processed_images.add(filename)
                else:
                    print("Could not stitch images:", combined_image_path, image_path)

                
            if len(processed_images) == len(filenames):
                break