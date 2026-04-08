import numpy as np

from tdt3d import ProjectionInput, TwoDToThreeDTool

axis_align = np.load("axis_align.npy")
K = np.load("intrinsic.npy")

views = [
    ProjectionInput(
        depth_image="depth1.png",
        intrinsic_matrix=K,
        extrinsic_matrix=np.load("extrinsic1.npy"),
        world_to_axis_align_matrix=axis_align,
        mask=np.load("mask1.npy"),
    ),
    ProjectionInput(
        depth_image="depth2.png",
        intrinsic_matrix=K,
        extrinsic_matrix=np.load("extrinsic2.npy"),
        world_to_axis_align_matrix=axis_align,
        mask=np.load("mask2.npy"),
    ),
]

tool = TwoDToThreeDTool()
points, bbox = tool.run_multi_view(views)

print("Fused points shape:", points.shape)
print("AABB [cx, cy, cz, dx, dy, dz]:", bbox)
