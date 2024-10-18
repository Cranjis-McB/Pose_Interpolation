# Pose Interpolation

Given the Cam-2-World poses (or poses_bounds.npy file), Creates Interpolated poses between them using Cartesian Trajectory.

# Demo

we use poses_bounds.npy file from "fern" scene in nerf_llff dataset and created interpolated poses. Once Interpolation is done, 
we rendered the frames using pretrained nerf for these interpolated poses. Rendered frames and corresponding videos can be found in rendered_frames folder.