import numpy as np
from tdt3d import TwoDToThreeDTool, PointFilterConfig, MorphologyConfig

tool = TwoDToThreeDTool(
    point_filter=PointFilterConfig(filter_type="none"),
    morphology=MorphologyConfig(
        erosion=False,
        dilation=False,
        kernel_size=1,
        keep_largest_components=False,
        num_components=1,
    ),
)

# 用大一點的 mask，不要用 2x2
mask = np.zeros((10, 10), dtype=bool)
mask[2:6, 3:8] = True

# depth in millimeters
depth_image = np.full((10, 10), 1000, dtype=np.uint16)  # 1000 mm = 1 m

intrinsic_matrix = np.eye(4, dtype=np.float64)
extrinsic_matrix = np.eye(4, dtype=np.float64)
world_to_axis_align_matrix = np.eye(4, dtype=np.float64)

points, bbox = tool.run_single_view(
    mask=mask,
    depth_image=depth_image,
    intrinsic_matrix=intrinsic_matrix,
    extrinsic_matrix=extrinsic_matrix,
    world_to_axis_align_matrix=world_to_axis_align_matrix,
    do_post_process=False,   # 這行很重要
)

print("points:")
print(points)
print("points shape:", points.shape)
print("bbox:")
print(bbox)