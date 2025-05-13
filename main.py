import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import open3d as o3d

# Path to your folders
rgb_folder = 'rgb/data'
depth_folder = 'depth/data'

# Camera intrinsic parameters (for Kinect or your camera model)
fx, fy, cx, cy = 525.0, 525.0, 319.5, 239.5  # These values are for Kinect


def load_rgb_and_depth(rgb_folder, depth_folder):
    rgb_images = sorted(os.listdir(rgb_folder))
    depth_images = sorted(os.listdir(depth_folder))
    rgb_frames = []
    depth_frames = []

    for rgb_image, depth_image in zip(rgb_images, depth_images):
        rgb_path = os.path.join(rgb_folder, rgb_image)
        depth_path = os.path.join(depth_folder, depth_image)

        rgb_frame = cv2.imread(rgb_path)
        if rgb_frame is None:
            print(f"Warning: Failed to load RGB image: {rgb_path}")
            continue

        depth_frame = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # Load as 16-bit
        if depth_frame is None:
            print(f"Warning: Failed to load depth image: {depth_path}")
            continue

        # Verify image dimensions and content
        if rgb_frame.size == 0 or depth_frame.size == 0:
            print(f"Warning: Empty image detected for {rgb_image} or {depth_image}")
            continue

        # Print image info for debugging
        print(
            f"Loaded RGB: {rgb_image}, shape={rgb_frame.shape}, dtype={rgb_frame.dtype}, range=[{np.min(rgb_frame)},{np.max(rgb_frame)}]")
        print(
            f"Loaded depth: {depth_image}, shape={depth_frame.shape}, dtype={depth_frame.dtype}, range=[{np.min(depth_frame)},{np.max(depth_frame)}]")

        rgb_frames.append(rgb_frame)
        depth_frames.append(depth_frame)

    print(f"Successfully loaded {len(rgb_frames)} RGB-D frames")
    return rgb_frames, depth_frames


def feature_matching(prev_img, curr_img):
    # Convert to grayscale if the images are in color
    if len(prev_img.shape) == 3:
        prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_img
        curr_gray = curr_img

    # Debug: visualize the grayscale images
    cv2.imwrite("debug_prev_gray.png", prev_gray)
    cv2.imwrite("debug_curr_gray.png", curr_gray)

    # Optional: Apply histogram equalization for better feature matching
    prev_gray = cv2.equalizeHist(prev_gray)
    curr_gray = cv2.equalizeHist(curr_gray)

    # Optional: Apply Gaussian blur to reduce noise
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)

    # Debug: Save processed images
    cv2.imwrite("debug_prev_processed.png", prev_gray)
    cv2.imwrite("debug_curr_processed.png", curr_gray)

    # Try different feature detection methods
    # Method 1: ORB (Fast but less accurate)
    orb = cv2.ORB_create(nfeatures=3000, scaleFactor=1.2, nlevels=8,
                         edgeThreshold=31, firstLevel=0, WTA_K=2,
                         patchSize=31, fastThreshold=20)

    prev_kp, prev_desc = orb.detectAndCompute(prev_gray, None)
    curr_kp, curr_desc = orb.detectAndCompute(curr_gray, None)

    # Debug: Draw keypoints on images
    prev_kp_img = cv2.drawKeypoints(prev_gray, prev_kp, None, color=(0, 255, 0),
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    curr_kp_img = cv2.drawKeypoints(curr_gray, curr_kp, None, color=(0, 255, 0),
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite("debug_prev_keypoints.png", prev_kp_img)
    cv2.imwrite("debug_curr_keypoints.png", curr_kp_img)

    print(f"Detected keypoints: prev={len(prev_kp)}, curr={len(curr_kp)}")

    # Check if we have enough features
    if prev_desc is None or curr_desc is None or len(prev_desc) < 8 or len(curr_desc) < 8:
        print("Not enough features detected. Trying SIFT...")

        # Method 2: SIFT (More robust but slower)
        try:
            sift = cv2.SIFT_create(nfeatures=3000)
            prev_kp, prev_desc = sift.detectAndCompute(prev_gray, None)
            curr_kp, curr_desc = sift.detectAndCompute(curr_gray, None)

            print(f"SIFT detected keypoints: prev={len(prev_kp)}, curr={len(curr_kp)}")

            if prev_desc is None or curr_desc is None or len(prev_desc) < 8 or len(curr_desc) < 8:
                print("Still not enough features with SIFT.")
                return np.array([]), np.array([])

            # Use FLANN matcher for SIFT features
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(prev_desc, curr_desc, k=2)

            # Apply Lowe's ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        except Exception as e:
            print(f"SIFT failed: {e}")
            return np.array([]), np.array([])
    else:
        # For ORB features, use Hamming distance
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(prev_desc, curr_desc)

        # Sort by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Select good matches
        max_distance = 64  # Adjust this threshold as needed
        good_matches = [m for m in matches if m.distance < max_distance]

    # Debug: Draw matches
    print(f"Total matches: {len(matches) if 'matches' in locals() else 0}, Good matches: {len(good_matches)}")

    if len(good_matches) >= 8:
        matches_img = cv2.drawMatches(prev_gray, prev_kp, curr_gray, curr_kp, good_matches[:30], None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite("debug_matches.png", matches_img)

    if len(good_matches) < 8:  # Need at least 8 points for findEssentialMat
        print("Not enough good matches for pose estimation.")
        return np.array([]), np.array([])

    # Extract matched point coordinates
    prev_pts = np.float32([prev_kp[m.queryIdx].pt for m in good_matches])
    curr_pts = np.float32([curr_kp[m.trainIdx].pt for m in good_matches])

    return prev_pts, curr_pts


def compute_camera_pose(prev_pts, curr_pts, prev_depth, curr_depth, K):
    # Check if we have enough points
    if len(prev_pts) < 8 or len(curr_pts) < 8:
        return np.eye(3), np.zeros((3, 1)), np.array([])

    # Calculate the essential matrix
    E, mask = cv2.findEssentialMat(curr_pts, prev_pts, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    if E is None or E.shape != (3, 3):
        return np.eye(3), np.zeros((3, 1)), np.array([])

    # Recover pose from the essential matrix
    _, R, t, mask = cv2.recoverPose(E, curr_pts, prev_pts, K)

    # Create projection matrices
    p1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))  # First camera projection matrix
    P2 = np.dot(K, np.hstack((R, t)))  # Second camera projection matrix

    # Format points for triangulation - cv2.triangulatePoints expects 2xN arrays
    prev_pts_2xn = prev_pts.T  # Shape becomes (2, N)
    curr_pts_2xn = curr_pts.T  # Shape becomes (2, N)

    # Triangulate points
    points_4d = cv2.triangulatePoints(p1, P2, prev_pts_2xn, curr_pts_2xn)

    # Convert from homogeneous coordinates to 3D
    points_3d = points_4d[:3] / points_4d[3]  # Normalize and keep only X, Y, Z
    points_3d = points_3d.T  # Convert to Nx3 format

    return R, t, points_3d


def depth_to_point_cloud(depth_frame, K):
    if depth_frame is None:
        print("Warning: Depth frame is None")
        return np.array([])

    # Check if depth frame has valid data
    if np.max(depth_frame) == 0 or np.min(depth_frame) == np.max(depth_frame):
        print(f"Warning: Invalid depth frame (min={np.min(depth_frame)}, max={np.max(depth_frame)})")
        return np.array([])

    # Get the dimensions of the depth frame
    h, w = depth_frame.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Create a mask for valid depth values
    # 0 typically means no depth data, and very large values might be noise
    min_depth = 100  # 0.1 meters (assuming depth is in mm)
    max_depth = 5000  # 5 meters (adjust based on your sensor's range)
    valid_mask = (depth_frame > min_depth) & (depth_frame < max_depth)

    # Calculate the number of valid depth points
    valid_points_count = np.sum(valid_mask)
    if valid_points_count < 100:  # Arbitrary threshold for minimum valid points
        print(f"Warning: Too few valid depth points ({valid_points_count})")
        return np.array([])

    # Apply the pinhole camera model only on valid points
    z = depth_frame[valid_mask].astype(float) / 1000.0  # Convert to meters
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    # Apply the pinhole camera model
    X = (x_valid - K[0, 2]) * z / K[0, 0]
    Y = (y_valid - K[1, 2]) * z / K[1, 1]

    # Stack the points
    points = np.vstack((X, Y, z)).T

    # Check for invalid values (NaN, Inf)
    valid_idx = np.all(np.isfinite(points), axis=1)
    points = points[valid_idx]

    # Remove outliers (points too far from the camera)
    dist = np.linalg.norm(points, axis=1)
    inlier_mask = dist < 5.0  # Keep points within 5 meters
    points = points[inlier_mask]

    # Downsample if necessary
    max_points = 5000
    if len(points) > max_points:
        # Random downsampling
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]

    print(f"Generated {len(points)} 3D points from depth")
    return points


# Real-time trajectory plot
def initialize_plot():
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('Camera Trajectory')
    ax.grid(True)
    return fig, ax


def update_trajectory_plot(ax, trajectory):
    if not trajectory:
        return

    trajectory = np.array(trajectory)
    ax.clear()
    ax.plot(trajectory[:, 0], trajectory[:, 2], 'b-', label="Trajectory")
    ax.plot(trajectory[0, 0], trajectory[0, 2], 'go', label="Start")
    ax.plot(trajectory[-1, 0], trajectory[-1, 2], 'ro', label="Current")
    ax.legend(loc="upper left")
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('Camera Trajectory')
    ax.grid(True)
    plt.draw()  # Refresh the plot
    plt.pause(0.001)  # Pause for a short time to allow updates



def visualize_point_cloud(points, window_name="Point Cloud"):
    if len(points) == 0:
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Add color for better visualization (optional)
    colors = np.zeros((len(points), 3))
    colors[:, 0] = np.ones(len(points)) * 0.5  # Red channel
    colors[:, 1] = np.ones(len(points)) * 0.5  # Green channel
    colors[:, 2] = np.ones(len(points)) * 0.8  # Blue channel
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=800, height=600)
    vis.add_geometry(pcd)

    # Set some rendering options
    opt = vis.get_render_option()
    opt.point_size = 1.0
    opt.background_color = np.array([0, 0, 0])  # Black background

    # Run the visualizer
    vis.run()
    vis.destroy_window()


def main():
    print("Starting SLAM pipeline...")

    # Check if folders exist
    if not os.path.exists(rgb_folder):
        print(f"Error: RGB folder '{rgb_folder}' does not exist")
        return
    if not os.path.exists(depth_folder):
        print(f"Error: Depth folder '{depth_folder}' does not exist")
        return

    # Create output folder for debug images
    debug_folder = "debug_output"
    os.makedirs(debug_folder, exist_ok=True)

    # Load the frames
    print(f"Loading frames from {rgb_folder} and {depth_folder}")
    rgb_frames, depth_frames = load_rgb_and_depth(rgb_folder, depth_folder)

    if not rgb_frames or not depth_frames:
        print("No frames loaded. Check your folder paths and image formats.")
        return

    print(f"Processing {len(rgb_frames)} frames")

    # Initialize camera intrinsic matrix
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    print(f"Camera matrix K:\n{K}")

    # For visualizing the trajectory
    trajectory = []
    cumulative_transform = np.eye(4)  # Start with identity transformation

    # Initialize the trajectory plot
    fig, ax = initialize_plot()

    # Accumulate point cloud - store as list of small point clouds for memory efficiency
    all_points = []

    # Sample interval - process every N-th frame for speed
    sample_interval = 5  # Adjust as needed

    # Process frames
    for i in range(1, len(rgb_frames), sample_interval):
        frame_idx = min(i, len(rgb_frames) - 1)
        print(f"\nProcessing frame pair {i}/{len(rgb_frames) - 1}")

        prev_rgb = rgb_frames[i - 1]
        curr_rgb = rgb_frames[frame_idx]
        prev_depth = depth_frames[i - 1]
        curr_depth = depth_frames[frame_idx]

        # Save sample images for debugging
        if i % 50 == 1:
            cv2.imwrite(os.path.join(debug_folder, f"rgb_frame_{i - 1}.png"), prev_rgb)

            # Visualize depth image (normalize for display)
            if prev_depth is not None:
                depth_norm = cv2.normalize(prev_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cv2.imwrite(os.path.join(debug_folder, f"depth_frame_{i - 1}.png"), depth_norm)
                print(
                    f"Depth stats - min: {np.min(prev_depth)}, max: {np.max(prev_depth)}, mean: {np.mean(prev_depth)}")

        # Match features between consecutive frames
        prev_pts, curr_pts = feature_matching(prev_rgb, curr_rgb)

        if len(prev_pts) < 8:
            print(f"Not enough matched features in frame {i}. Skipping...")
            # Use identity transform for this frame
            trajectory.append(cumulative_transform[:3, 3])
            update_trajectory_plot(ax, trajectory)
            continue

        # Estimate camera pose (rotation and translation)
        R, t, points_3d = compute_camera_pose(prev_pts, curr_pts, prev_depth, curr_depth, K)

        if points_3d.size == 0:
            print(f"Failed to compute pose for frame {i}. Skipping...")
            # Use identity transform for this frame
            trajectory.append(cumulative_transform[:3, 3])
            update_trajectory_plot(ax, trajectory)
            continue

        # Motion validation (simple check for unreasonable motion)
        translation_magnitude = np.linalg.norm(t)
        if translation_magnitude > 1.0:  # More than 1 meter between consecutive frames seems unreasonable
            print(f"Warning: Large translation detected ({translation_magnitude:.2f}m). Scaling down.")
            t = t * (1.0 / translation_magnitude)  # Normalize to 1m max

        # Update the cumulative transformation
        current_transform = np.eye(4)
        current_transform[:3, :3] = R
        current_transform[:3, 3] = t.flatten()

        # Apply the transform with a small damping factor for stability
        damping = 0.5  # Adjust between 0 and 1
        interpolated_transform = np.eye(4)
        interpolated_transform[:3, :3] = R * damping + np.eye(3) * (1 - damping)
        interpolated_transform[:3, 3] = t.flatten() * damping

        cumulative_transform = cumulative_transform @ interpolated_transform

        # Extract the current position (translation) from the cumulative transform
        current_position = cumulative_transform[:3, 3]
        trajectory.append(current_position)

        # Update the trajectory plot
        update_trajectory_plot(ax, trajectory)

        # Convert depth to point cloud and transform to world coordinates
        if i % 20 == 1:  # Only add points every 20 frames to reduce memory usage
            current_pc = depth_to_point_cloud(prev_depth, K)

            if len(current_pc) > 0:
                # Transform point cloud to world coordinates
                homogeneous_points = np.hstack((current_pc, np.ones((len(current_pc), 1))))
                transformed_points = (cumulative_transform @ homogeneous_points.T).T[:, :3]

                # Downsample and append to global point cloud
                if len(transformed_points) > 500:
                    indices = np.random.choice(len(transformed_points), 500, replace=False)
                    transformed_points = transformed_points[indices]

                all_points.append(transformed_points)

        # Show progress every 10 frames
        if i % 10 == 0:
            print(f"Current position: {current_position}")

            # Save the current trajectory to file
            if trajectory:
                trajectory_array = np.array(trajectory)
                np.savetxt("trajectory.txt", trajectory_array)
                print(f"Saved trajectory with {len(trajectory)} points")

    plt.ioff()  # Turn off interactive mode
    plt.show()  # Keep the trajectory plot visible

    # Visualize the final point cloud
    if all_points:
        combined_point_cloud = np.vstack(all_points)
        print(f"Final point cloud has {len(combined_point_cloud)} points")

        # Filter outliers
        mean = np.mean(combined_point_cloud, axis=0)
        std = np.std(combined_point_cloud, axis=0)
        distance_from_mean = np.linalg.norm(combined_point_cloud - mean, axis=1)
        max_distance = 3 * np.mean(std)  # 3 sigma
        inlier_mask = distance_from_mean < max_distance
        filtered_point_cloud = combined_point_cloud[inlier_mask]

        print(f"After filtering: {len(filtered_point_cloud)} points")

        # Save point cloud to file
        np.savetxt("point_cloud.txt", filtered_point_cloud)
        print("Saved point cloud to point_cloud.txt")

        # Visualize
        visualize_point_cloud(filtered_point_cloud, "Final Point Cloud")
    else:
        print("No valid points were collected.")


if __name__ == "__main__":
    main()