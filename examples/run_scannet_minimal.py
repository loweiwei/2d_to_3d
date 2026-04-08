import cv2
import numpy as np
import open3d as o3d
from tdt3d.core import TwoDToThreeDTool, PointFilterConfig

scene_id = "scene0207_00"
image_id = "00000"

rgb_path = f"{scene_id}/{image_id}.jpg"
depth_path = f"{scene_id}/{image_id}.png"
extrinsic_path = f"{scene_id}/{image_id}.txt"
intrinsic_path = f"{scene_id}/intrinsic.txt"

rgb = cv2.imread(rgb_path)
if rgb is None:
    raise FileNotFoundError(rgb_path)

depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
if depth_image is None:
    raise FileNotFoundError(depth_path)

extrinsic_matrix = np.loadtxt(extrinsic_path).astype(np.float64)
intrinsic_matrix = np.loadtxt(intrinsic_path).astype(np.float64)

if extrinsic_matrix.shape != (4, 4):
    raise ValueError(f"extrinsic_matrix shape error: {extrinsic_matrix.shape}")

if intrinsic_matrix.shape != (4, 4):
    raise ValueError(f"intrinsic_matrix shape error: {intrinsic_matrix.shape}")

world_to_axis_align_matrix = np.eye(4, dtype=np.float64)

H, W = rgb.shape[:2]

print("rgb shape:", rgb.shape)
print("depth shape:", depth_image.shape, depth_image.dtype)
print("depth min/max:", depth_image.min(), depth_image.max())
print("extrinsic shape:", extrinsic_matrix.shape)
print("intrinsic shape:", intrinsic_matrix.shape)

x, y, w, h = cv2.selectROI("Select ROI", rgb, showCrosshair=True, fromCenter=False)
cv2.destroyAllWindows()

if w == 0 or h == 0:
    raise ValueError("No ROI selected")

mask = np.zeros((H, W), dtype=np.uint8)
mask[int(y):int(y+h), int(x):int(x+w)] = 1

print("mask sum:", mask.sum())

mask_depth = cv2.resize(
    mask.astype(np.uint8),
    (depth_image.shape[1], depth_image.shape[0]),
    interpolation=cv2.INTER_NEAREST,
)
print("masked nonzero depth:", np.count_nonzero(depth_image[mask_depth > 0]))

tool = TwoDToThreeDTool(
    point_filter=PointFilterConfig(filter_type="none"),
    depth_scale=0.001,
)

points, bbox = tool.run_single_view(
    mask=mask,
    depth_image=depth_image,
    intrinsic_matrix=intrinsic_matrix,
    extrinsic_matrix=extrinsic_matrix,
    world_to_axis_align_matrix=world_to_axis_align_matrix,
    color_image=rgb,
    do_post_process=True,
)

print("points shape:", points.shape)
print("bbox:", bbox)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])

o3d.visualization.draw_geometries([pcd], window_name="3D Point Cloud")