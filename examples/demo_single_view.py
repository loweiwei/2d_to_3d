import numpy as np

from tdt3d import TwoDToThreeDTool

# Replace these with your own files / arrays.
mask = np.load("mask.npy")
depth_image = "depth.png"
intrinsic_matrix = np.load("intrinsic.npy")
extrinsic_matrix = np.load("extrinsic.npy")
world_to_axis_align_matrix = np.load("axis_align.npy")


tool = TwoDToThreeDTool()
points, bbox = tool.run_single_view(
    mask=mask,
    depth_image=depth_image,
    intrinsic_matrix=intrinsic_matrix,
    extrinsic_matrix=extrinsic_matrix,
    world_to_axis_align_matrix=world_to_axis_align_matrix,
)

print("Projected points shape:", points.shape)
print("AABB [cx, cy, cz, dx, dy, dz]:", bbox)
