import os
import cv2
import time
from transform_getter import TransformGetter
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import g2o
import numpy as np

def add_relative_motion_to_pose_graph(optimizer, relative_motion):
    # Create a new vertex for the current pose
    vertex = g2o.VertexSE3Expmap()
    
    # Set the estimate for the vertex to the current pose
    vertex.set_estimate(relative_motion)
    
    # Add the vertex to the optimizer
    optimizer.add_vertex(vertex)
    
    # Create a new edge for the relative motion
    edge = g2o.EdgeSE3Expmap()
    
    # Set the vertices for the edge to the previous and current pose
    edge.set_vertex(0, optimizer.vertex(optimizer.vertices().size() - 2))
    edge.set_vertex(1, optimizer.vertex(optimizer.vertices().size() - 1))
    
    # Set the measurement for the edge to the relative motion
    edge.set_measurement(relative_motion)
    
    # Add the edge to the optimizer
    optimizer.add_edge(edge)

def add_loop_closure_to_pose_graph(optimizer, loop_closure_motion, loop_closure_vertex_id):
    # Create a new edge for the loop closure
    edge = g2o.EdgeSE3Expmap()

    # Set the vertices for the edge to the current pose and the loop closure pose
    edge.set_vertex(0, optimizer.vertex(optimizer.vertices().size() - 1))  # current pose
    edge.set_vertex(1, optimizer.vertex(loop_closure_vertex_id))  # loop closure pose

    # Set the measurement for the edge to the loop closure motion
    edge.set_measurement(loop_closure_motion)

    # Add the edge to the optimizer
    optimizer.add_edge(edge)

def run_slam(image,previous_image_filename=None, current_image_filename=None):

    transform_getter = TransformGetter('/home/yi/Documents/EBB-CV/feature_detection/SimData_2023-6-13/EEB_Transform.txt')

    # Set the loop closure threshold
    loop_closure_threshold = 0.5

    # Initialize the ORB detector
    orb = cv2.ORB_create()

    # Initialize the BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Initialize the pose graph
    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(solver)

    # Initialize the bag-of-words model
    bow = KMeans(n_clusters=1000)

    # Initialize the place recognizer
    recognizer = NearestNeighbors(n_neighbors=1)

    # Initialize some variables
    prev_image = cv2.imread(previous_image_filename, cv2.IMREAD_GRAYSCALE)
    prev_keypoints, prev_descriptors = orb.detectAndCompute(prev_image, None)

    # Initialize the initial pose  
    prev_timestamp = previous_image_filename.split('_')[2] + '_' + previous_image_filename.split('_')[3].replace('.jpg', '')
    prev_transform = transform_getter.get_transform(prev_timestamp)
    prev_x, _, prev_y = prev_transform.position
    _, yaw_degrees1, _ = prev_transform.rotation
    prev_pose = create_pose([prev_x, prev_y, 0], yaw_degrees1)

    # Initialize the database of visual words and poses
    visual_words = []
    poses = []

    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(image, None)

    # Update the bag-of-words model and the database of visual words
    bow.partial_fit(descriptors)
    visual_words.append(bow.predict(descriptors))
    poses.append(prev_pose)

    # Update the place recognizer
    recognizer.fit(visual_words)

    if prev_image is not None:
        # Match features between the current and previous image
        matches = bf.match(prev_descriptors, descriptors)

        # Estimate the relative motion between the current and previous image
        relative_motion = estimate_relative_motion(previous_image_filename, current_image_filename)

        print('Relative motion estimated')

        # Add the relative motion to the pose graph
        add_relative_motion_to_pose_graph(optimizer, relative_motion)

        print('Relative motion added to the pose graph')

        # Check for loop closure
        dist, ind = recognizer.kneighbors(visual_words[-1], n_neighbors=1)
        if dist < loop_closure_threshold:
            # A loop closure is detected
            loop_closure_pose = poses[ind[0]]

            # Estimate the relative motion between the current image and the loop closure
            loop_closure_motion = estimate_relative_motion(current_image_filename, loop_closure_pose)

            # Add the loop closure to the pose graph
            add_loop_closure_to_pose_graph(optimizer, loop_closure_motion)

    # Remember the current image, keypoints, descriptors, and pose for the next iteration
    prev_image = image
    # prev_keypoints = keypoints
    prev_descriptors = descriptors
    # prev_pose = initial_pose

    # Optimize the pose graph
    optimizer.initialize_optimization()
    optimizer.optimize(10)  # Perform 10 iterations of optimization

    # Now the poses in the pose graph have been corrected by the optimization
    # You can use these poses to construct your map
    return poses[-1]



def create_pose(position, yaw_degrees):
    # Convert the yaw angle from degrees to radians
    yaw_radians = np.radians(yaw_degrees)

    # Create the rotation matrix for the yaw
    R = np.array([
        [np.cos(yaw_radians), -np.sin(yaw_radians), 0],
        [np.sin(yaw_radians),  np.cos(yaw_radians), 0],
        [0, 0, 1]
    ])

    # Create the pose
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = position

    return pose

def estimate_relative_motion(image1_filename, image2_filename):

    transform_getter = TransformGetter('/home/yi/Documents/EBB-CV/feature_detection/SimData_2023-6-13/EEB_Transform.txt')

    timestamp1 = image1_filename.split('_')[2] + '_' + image1_filename.split('_')[3].replace('.jpg', '')
    timestamp2 = image2_filename.split('_')[2] + '_' + image2_filename.split('_')[3].replace('.jpg', '')

    # Get the transform for this timestamp
    transform1 = transform_getter.get_transform(timestamp1)
    transform2 = transform_getter.get_transform(timestamp2)

    # Extract the position and yaw from the transform
    x1, _, y1 = transform1.position
    _, yaw_degrees1, _ = transform1.rotation

    x2, _, y2 = transform2.position
    _, yaw_degrees2, _ = transform2.rotation

    # Create the pose
    pose1 = create_pose([x1, y1, 0], yaw_degrees1)
    pose2 = create_pose([x2, y2, 0], yaw_degrees2)

    # Calculate the relative motion
    relative_motion = np.linalg.inv(pose1) @ pose2

    print('Relative motion from {} to {}:'.format(timestamp1, timestamp2))

    return relative_motion


if __name__ == '__main__':
    # Directory containing the images
    image_dir = "feature_detection/SimData_2023-6-13"

    transform_getter = TransformGetter('/home/yi/Documents/EBB-CV/feature_detection/SimData_2023-6-13/EEB_Transform.txt')
    # Create a set to store the names of processed images
    processed_images = set()

    # Initialize some variables
    prev_image = None
    prev_pose = None
    visual_words = []
    poses = []

    # Loop over the sorted list of filenames
    while True:
        filenames = os.listdir(image_dir)
        filenames = list(filter(lambda filename: filename.endswith('.jpg'), filenames))
        filenames.sort()

        if len(filenames) == 1:
            # Only one image in the directory, so we can't do anything yet
            time.sleep(0.1)
            print('Waiting for more images...')
            continue

        else:
            for filename in filenames:
                if filename.endswith('.jpg') and filename not in processed_images:
                    # Extract the timestamp from the filename
                    current_image_filename = filename
                    prev_image_filename = filenames[filenames.index(filename) - 2]

                    # Load the image
                    img = cv2.imread(os.path.join(image_dir, filename))

                    # Run SLAM on the image
                    pose = run_slam(img, prev_image_filename, current_image_filename)
                    # Do something with the pose...
                    print(pose)

                    processed_images.add(filename)
                    prev_image_filename = current_image_filename
                    prev_image = img
                    prev_pose = pose

        # Wait for a short while before checking for new images
        time.sleep(0.1)
