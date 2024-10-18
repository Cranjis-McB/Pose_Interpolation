# Imports
import datetime
import time
import traceback
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R, Slerp


def get_extrinsics_and_intrinsics(poses_bounds: np.ndarray):
    poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
    depth_bounds = poses_bounds[:, 15:]
    intrinsics = poses[0, :, -1]
    extrinsics = poses[:, :, :4]
    # Make it homogeneous
    rows_to_add = np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), extrinsics.shape[0], axis=0)  # Shape (12, 1, 4)
    extrinsics_homo = np.concatenate((extrinsics, rows_to_add), axis=1)
    return extrinsics_homo, intrinsics, depth_bounds # Extrinsics and Intrinsics


def llff2opencv(transformation_matrix: np.ndarray):
    """
    :param transformation_matrix: 4 x 4 Homogeneous Transformation Matrix, [R|T] of the poses_bounds.npy -- C2W and LLFF Convention
    :return: 4 x 4 Homogeneous Transformation Matrix --W2C and OpenCV Convention
    """

    # Step-1: Correct the Convention [-y, x, -z]  => [x, -y, -z]
    transformation_matrix_cv = np.concatenate([
        transformation_matrix[:, :, 1:2],
        transformation_matrix[:, :,  0:1],
        -transformation_matrix[:, :, 2:3],
        transformation_matrix[:, :, 3:]
    ],
        -1
    )

    # Step-2: C2W => W2C Convention
    transformation_matrix_cv = np.linalg.inv(transformation_matrix_cv)

    return transformation_matrix_cv


def opencv2llff(transformation_matrix: np.ndarray):
    """
    Convert W2C to C2W
    """

    # Step-1: W2C => C2W Convention
    transformation_matrix = np.linalg.inv(transformation_matrix)

    # Step-2: Correct the Convention  [x, -y, -z] => [-y, x, -z]
    transformation_matrix_llff = np.concatenate([
        transformation_matrix[:, :, 1:2],
        transformation_matrix[:, :, 0:1],
        -transformation_matrix[:, :, 2:3],
        transformation_matrix[:, :, 3:]
    ],
        -1
    )

    return transformation_matrix_llff


def compute_cartesian_trajectory(T1, T2, D1, D2, n):
    """
    Compute a Cartesian trajectory between two homogeneous transformations T1 and T2.

    Args:
        T1 (numpy.ndarray): Initial 4x4 homogeneous transformation matrix.
        T2 (numpy.ndarray): Final 4x4 homogeneous transformation matrix.
        D1: Initial Depth Bounds
        D2: Final Depth Bounds
        n (int): Number of steps in the trajectory.

    Returns:
        list: List of n homogeneous transformation matrices interpolating from T1 to T2.
    """
    # Extract translation vectors
    t1 = T1[:3, 3]
    t2 = T2[:3, 3]

    # Extract rotation matrices
    R1 = T1[:3, :3]
    R2 = T2[:3, :3]

    # Create Rotation objects from the rotation matrices
    R_start = R.from_matrix(R1)
    R_end = R.from_matrix(R2)

    # Define the keyframe rotations
    key_times = [0, 1]  # Start and end times for interpolation
    key_rots = R.from_matrix([R1, R2])  # Rotation matrices for SLERP

    # Create the Slerp object for interpolation
    slerp = Slerp(key_times, key_rots)

    # Initialize list to hold trajectory transforms
    traj = []
    depth_traj = []

    for i in range(n):
        alpha = i / (n - 1)  # Interpolation parameter

        # Interpolate translation
        t_interp = t1 + (t2 - t1) * alpha

        # Interpolate Depth bounds
        d_interpt = D1 + (D2 - D1) * alpha

        # Interpolate rotation using SLERP
        R_interp = slerp([alpha]).as_matrix()[0]

        # Construct the homogeneous transformation matrix
        T_interp = np.eye(4)
        T_interp[:3, :3] = R_interp  # Set rotation part
        T_interp[:3, 3] = t_interp  # Set translation part

        # Append to the trajectory list
        traj.append(T_interp)
        depth_traj.append(d_interpt)

    return traj, depth_traj


def pose_interpolation(poses_bounds_path, N = 3):
    """
    Given C2W Cam poses and Number of Interpolation N bw 2 consecutive poses,
    creates interpolated poses.
    :return: Interpolated Poses_bounds.npy file
    """
    # Load Poses_bounds.npy file
    poses_bounds = np.load(poses_bounds_path)

    # Get Poses, Intrinsics, and Depth bounds
    extrinsics_c2w, intrinsics, depth_bounds = get_extrinsics_and_intrinsics(poses_bounds)

    # Convert Poses to W2C Format
    extrinsics_w2c = llff2opencv(extrinsics_c2w)

    # Create Interpolated Poses
    interpolated_poses = []
    interpolated_depths = []

    num_poses  = extrinsics_w2c.shape[0]
    for index in range(num_poses - 1):
        T1, T2 = extrinsics_w2c[index], extrinsics_w2c[index + 1]
        D1, D2 = depth_bounds[index], depth_bounds[index + 1]
        traj, depth_traj = compute_cartesian_trajectory(T1, T2, D1, D2, N + 2)  # N + 2 (for including 1st and last poses)
        # Avoid duplicate poses
        if index > 0:
            # Dont include final pose/bound
            traj = traj[1:]
            depth_traj = depth_traj[1:]

        interpolated_poses += traj
        interpolated_depths += depth_traj

    interpolated_poses = np.array(interpolated_poses)
    interpolated_depths = np.array(interpolated_depths)

    # Convert the poses to C2W again
    interpolated_poses_c2w = opencv2llff(interpolated_poses)

    # Save Poses, Intrinsics, and Depths into poses_bounds.npy format
    N_images = interpolated_poses_c2w.shape[0]
    intrinsics_expanded = np.tile(intrinsics, (N_images, 1))
    intrinsics_expanded = intrinsics_expanded.reshape(N_images, 3, 1)
    poses_interpolated = np.concatenate([interpolated_poses_c2w[:, 0:3, 0:4], intrinsics_expanded], axis=2)
    poses_interpolated = poses_interpolated.reshape(N_images, 15)
    poses_bounds_interpolated = np.concatenate([poses_interpolated, interpolated_depths], axis=1)

    return poses_bounds_interpolated


def main():

    SCENES = ["fern"] #, "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]

    for scene in SCENES:
        poses_bounds_path = Path(f"data/nerf_llff_data/{scene}/poses_bounds.npy")
        N = 3 # 3 interpolations bw two consecutive poses.

        # Pose Interpolation
        poses_bounds_interpolated = pose_interpolation(poses_bounds_path, N=N)

        # Save
        save_dir = Path(f"data/nerf_llff_data/{scene}/poses_bounds_interpolated.npy")
        np.save(save_dir, poses_bounds_interpolated)

    pass


if __name__ == '__main__':
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    try:
        main()
    except Exception as e:
        print(e)
        traceback.print_exc()
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))