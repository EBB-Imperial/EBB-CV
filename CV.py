import cv2
import numpy as np
import os
import shutil
from rotation import rotate_image
from transform_getter import TransformGetter
from perspective import ipm
from perspectivetest import perspective_transform
from combine_images import combine_images  # Import the function for combining images

def crop_to_nontransparent(img):
    # Find the bounding box of the non-transparent region
    y_nonzero, x_nonzero, _ = np.nonzero(img)
    min_x = np.min(x_nonzero)
    max_x = np.max(x_nonzero)
    min_y = np.min(y_nonzero)
    max_y = np.max(y_nonzero)

    # Crop the image to the bounding box
    return img[min_y:max_y, min_x:max_x]

# Load the transform data
transform_getter = TransformGetter('SimData_2023-6-13/EEB_Transform.txt')

# Directory containing the images
image_dir = '/home/yi/Documents/EBB-CV/SimData_2023-6-13'

# Directory to save the rotated images
rotated_dir = 'rotated_images'

# Directory to save the erspective Transform images
perspective_dir = 'perspective_images'

# Directory to save the combined images
combined_dir = 'existing_map'

# Delete the rotated images directory if it exists
if os.path.exists(rotated_dir):
    shutil.rmtree(rotated_dir)

# Create the rotated images directory
os.makedirs(rotated_dir)

# Delete the perspective images directory if it exists
if os.path.exists(perspective_dir):
    shutil.rmtree(perspective_dir)
    
# Create the perspective images directory
os.makedirs(perspective_dir)

# Delete the combined images directory if it exists
if os.path.exists(combined_dir):
    shutil.rmtree(combined_dir)

# Create the combined images directory
os.makedirs(combined_dir)

# Create a set to store the names of processed images
processed_images = set()

# Get a list of filenames and sort it
filenames = os.listdir(image_dir)
filenames = list(filter(lambda filename: filename.endswith('.jpg'), filenames))
filenames.sort()

# Initialize the combined image as an empty image of the same size as the first image
first_image_filename = filenames[0]
first_image = cv2.imread(os.path.join(image_dir, first_image_filename))
combined_image = np.zeros_like(first_image)

# Define FOV
FOV = 70

# Define height of the robot
height = 3.69

# Define angle of the camera
angle = 53.7

# Loop over the sorted list of filenames
while True:
    
    for filename in filenames:
        # Check if the file is a .jpg image and has not been processed
        if filename.endswith('.jpg') and filename not in processed_images:
            # Extract the timestamp from the filename
            timestamp = filename.split('_')[2] + '_' + filename.split('_')[3].replace('.jpg', '')
            
            # Load the image
            img = cv2.imread(os.path.join(image_dir, filename))

            # Convert the image to RGBA
            img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

            # Apply perspective transform
            img_transformed = perspective_transform(img_rgba)

            # Save the transformed image
            cv2.imwrite(os.path.join(perspective_dir, filename.replace('.jpg', '.png')), img_transformed)
            
            # Get the transform for the image's timestamp
            transform = transform_getter.get_transform(timestamp)
            
            # Rotate the image
            img_rotated = rotate_image(img_transformed, transform.rotation[1])

            # Crop the image to the non-transparent region
            img_cropped = crop_to_nontransparent(img_rotated)
            
            # Save the cropped image
            cv2.imwrite(os.path.join(rotated_dir, filename.replace('.jpg', '.png')), img_cropped)

            # Combine the cropped image with the combined image
            # combined_image = combine_images(combined_image, img_cropped)

            # Save the combined image
            # combined_image_filename = 'existing_map_' + str(len(processed_images) + 1) + '.png'
            # cv2.imwrite(os.path.join(combined_dir, combined_image_filename), combined_image)

            # Add the filename to the set of processed images
            processed_images.add(filename)
